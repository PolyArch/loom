#include "PnR/System/SystemPnrSearchDomain.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/MemoryConsistencyContract.h"
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
#include <optional>
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

struct FramedFabricDomain final {
  std::size_t countOffset = 0;
  std::uint64_t count = 0;
  std::vector<FramedRange> entries;
};

struct FramedServiceEntry final {
  FramedRange bytes;
  std::optional<FramedFabricDomain> targetDomain;
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

FramedFabricDomain locateFabricDomain(FramingCursor &cursor) {
  FramedFabricDomain domain;
  domain.countOffset = cursor.position();
  domain.count = cursor.u64();
  domain.entries.reserve(domain.count);
  for (std::uint64_t index = 0; index < domain.count; ++index) {
    const std::size_t begin = cursor.position();
    cursor.skipSizedBytes();
    domain.entries.push_back({begin, cursor.position()});
  }
  return domain;
}

void skipFabricDomain(FramingCursor &cursor) {
  (void)locateFabricDomain(cursor);
}

void skipRootDomain(FramingCursor &cursor) {
  const std::uint64_t count = cursor.u64();
  for (std::uint64_t index = 0; index < count; ++index)
    cursor.skipRootReference();
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

void skipAtomDomains(FramingCursor &cursor) {
  const std::uint32_t presence = cursor.u32();
  require((presence & ~0xfu) == 0,
          "test target-domain framing has unknown fields");
  if ((presence & (1u << 0)) != 0)
    skipFabricDomain(cursor);
  if ((presence & (1u << 1)) != 0)
    skipRootDomain(cursor);
  if ((presence & (1u << 2)) != 0)
    skipFabricDomain(cursor);
  if ((presence & (1u << 3)) != 0)
    skipFabricDomain(cursor);
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
      skipAtomDomains(cursor);
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
    const std::uint32_t targetKind = cursor.u32();
    require(targetKind <= 2,
            "test service target-domain kind is not canonical");
    if (targetKind != 0)
      entry.targetDomain = locateFabricDomain(cursor);
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

::fabric::MemoryActorContractDomain
fenceActorDomain(dataflow::AtomicOrdering ordering) {
  return take(::fabric::MemoryActorContractDomain::create(
      dataflow::OperationSchemaId::DataflowFence,
      {::fabric::MemoryActorContractClause(::fabric::FenceContractClause{
          {ordering}, {{dataflow::SyncScopeKind::System, {}, {}}}})}));
}

loom::adg::FinalizedFabricDesign
buildFenceFabric(const loom::ArtifactStore &store,
                 const loom::fabric::FinalizedFabricRoot &base) {
  const auto baseSystem = take(loom::fabric::requireSystemRoot(base.view()));
  require(!baseSystem.artifact().systemMemoryServices().empty(),
          "builtin System has no memory service for the fence fixture");
  const auto *memoryContract = baseSystem.memoryService(
      baseSystem.artifact().systemMemoryServices().front());
  require(memoryContract,
          "builtin System memory service has no canonical contract");
  require(!baseSystem.transportResources().empty(),
          "builtin System has no transport resource for the fence fixture");
  const auto *resourceContract = baseSystem.artifact().resourceContract(
      loom::fabric::FabricInventoryOwnerRef::of(
          baseSystem.transportResources().front()));
  require(resourceContract,
          "builtin System transport has no resource contract");

  auto module = take(loom::fabric::importEntireFabricRoot(
      base.directDependencies().front().root, store));
  loom::adg::DesignBuilder design(store);
  auto system = take(loom::adg::expandBuiltinSystem(
      design, loom::adg::BuiltinTargetPreset::Small, module));
  auto clock = take(system.createHardwareDomain());
  auto rate = take(system.createServiceRate(
      clock, 1, 1, 4,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>)));
  const auto bits128 = take(loom::adg::PortType::bits(128));
  auto transport = take(
      system.addTransportResource({{bits128}, {bits128}, *resourceContract}));
  auto pattern = take(system.addTransferPattern(transport, 0, {0}, 0));
  auto input = take(transport.input(0));
  auto output = take(transport.output(0));

  std::vector<loom::adg::HardwareDomainMember> clockMembers{
      transport.domainMember(), pattern.domainMember()};
  for (dataflow::AtomicOrdering ordering :
       {dataflow::AtomicOrdering::SeqCst, dataflow::AtomicOrdering::SeqCst,
        dataflow::AtomicOrdering::Acquire}) {
    auto memory = take(system.addMemoryService(*memoryContract));
    auto consistency = take(system.createHardwareDomain());
    const loom::fabric::MemoryConsistencyDomainRef consistencyRef(
        consistency.reference());
    auto domain = take(loom::fabric::FenceCapabilityDomain::create(
        fenceActorDomain(ordering), consistencyRef));
    auto initiateCapability =
        take(loom::fabric::CanonicalServiceCapabilityRecord::create(
            dataflow::semantics::ServiceKind::MemoryFence,
            loom::fabric::CanonicalServiceEndpointRole::Initiate, domain,
            rate));
    auto serveCapability =
        take(loom::fabric::CanonicalServiceCapabilityRecord::create(
            dataflow::semantics::ServiceKind::MemoryFence,
            loom::fabric::CanonicalServiceEndpointRole::Serve,
            std::move(domain), rate));
    auto initiateSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
        {std::move(initiateCapability)}));
    auto serveSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
        {std::move(serveCapability)}));
    auto initiate = take(system.addServiceEndpoint(memory, initiateSet));
    auto serve = take(system.addServiceEndpoint(memory, serveSet));
    auto initiateMemory = take(initiate.memory());
    auto serveMemory = take(serve.memory());

    if (llvm::Error error = system.attachServiceLegCarriers(
            initiateMemory, dataflow::semantics::ServiceKind::MemoryFence, 0,
            {output}))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.attachServiceLegCarriers(
            initiateMemory, dataflow::semantics::ServiceKind::MemoryFence, 1,
            {input}))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.attachServiceLegCarriers(
            serveMemory, dataflow::semantics::ServiceKind::MemoryFence, 0,
            {input}))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.attachServiceLegCarriers(
            serveMemory, dataflow::semantics::ServiceKind::MemoryFence, 1,
            {output}))
      fail(llvm::toString(std::move(error)));

    auto consistencyContract = take(::fabric::MemoryConsistencyContract::create(
        ::fabric::MemoryConsistencyContractDeclaration{
            {::fabric::MemoryConsistencyParticipant::service(
                loom::fabric::FabricMemoryServiceRef::system(
                    memory.reference()))},
            ::fabric::ReleaseVisibilityPoint::AtLinearization,
            ::fabric::MemoryConsistencyProgress(
                std::in_place_type<::fabric::FairEventual>),
            *resourceContract}));
    if (llvm::Error error =
            consistency.close({memory.domainMember(), initiate.domainMember(),
                               serve.domainMember()},
                              std::move(consistencyContract)))
      fail(llvm::toString(std::move(error)));
    clockMembers.push_back(memory.domainMember());
    clockMembers.push_back(initiate.domainMember());
    clockMembers.push_back(serve.domainMember());
  }

  auto clockContract =
      take(loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = clock.close(clockMembers, std::move(clockContract)))
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
          if (candidate.kind() !=
              dataflow::semantics::ServiceKind::MessageTransfer)
            return false;
          const auto *domain =
              std::get_if<loom::fabric::MessageTransferCapabilityDomain>(
                  &candidate.domain());
          return domain &&
                 llvm::any_of(domain->payloadTypes(),
                              [](mlir::Type t) { return t.isInteger(32); });
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

std::vector<loom::fabric::MemoryConsistencyDomainRef>
matchingFenceConsistencyDomains(
    const loom::fabric::FabricSystemRootView &system,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::pnr::SystemSearchServiceDomain &service) {
  const auto &operation =
      std::get<loom::mapping::OperationServiceObligationFamilyKey>(service.key);
  const auto family = std::get<dataflow::FenceActorFamilyRef>(operation);
  auto actorView = take(dataflow.resolve(family.actor));
  auto actor =
      take(dataflow::projectRegisteredActorSchemaProjection(actorView.op));

  std::vector<loom::fabric::MemoryConsistencyDomainRef> result;
  for (const auto endpoint : system.artifact().systemServiceEndpoints()) {
    const auto *capabilities = system.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() !=
            loom::fabric::CanonicalServiceEndpointPlane::Memory ||
        capabilities->role() !=
            loom::fabric::CanonicalServiceEndpointRole::Serve)
      continue;
    for (const auto &capability : capabilities->capabilities()) {
      const auto *fence = std::get_if<loom::fabric::FenceCapabilityDomain>(
          &capability.domain());
      if (capability.kind() == dataflow::semantics::ServiceKind::MemoryFence &&
          fence && fence->actorContracts().contains(actor))
        result.push_back(fence->consistencyDomain());
    }
  }
  llvm::sort(result, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::vector<loom::fabric::FabricTransportEndpointRef>
attachmentCarriers(const loom::fabric::FabricSystemRootView &system,
                   dataflow::StructuralOrdinal legOrdinal, bool source) {
  std::vector<loom::fabric::FabricTransportEndpointRef> carriers;
  auto direction = take(dataflow::semantics::getCanonicalServiceLegDirection(
      dataflow::semantics::ServiceKind::MemoryRead, legOrdinal));
  for (const loom::fabric::ServiceLegCarrierAttachmentRecord &attachment :
       system.serviceLegCarrierAttachments()) {
    if (attachment.kind() != dataflow::semantics::ServiceKind::MemoryRead ||
        attachment.legOrdinal() != legOrdinal)
      continue;
    const auto endpoint = std::get<loom::fabric::SystemServiceEndpointRef>(
        attachment.endpoint().owner.payload);
    const auto *capabilities = system.serviceEndpointCapabilities(endpoint);
    require(capabilities, "attachment endpoint has no capability set");
    const bool endpointIsInitiator =
        capabilities->role() ==
        loom::fabric::CanonicalServiceEndpointRole::Initiate;
    const bool initiatorIsSource =
        direction ==
        dataflow::semantics::ServiceLegDirection::InitiatorToServer;
    if (source ? endpointIsInitiator != initiatorIsSource
               : endpointIsInitiator == initiatorIsSource)
      continue;
    carriers.insert(carriers.end(), attachment.carriers().begin(),
                    attachment.carriers().end());
  }
  llvm::sort(carriers, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  carriers.erase(std::unique(carriers.begin(), carriers.end()), carriers.end());
  return carriers;
}

const loom::pnr::SystemSearchTransferTerminalDomain &
terminal(const loom::pnr::SystemSearchServiceDomain &service,
         dataflow::StructuralOrdinal legOrdinal, bool source) {
  const auto found =
      llvm::find_if(service.transferTerminals, [&](const auto &candidate) {
        if (const auto *key =
                std::get_if<loom::pnr::SystemTransferSourceTerminalKey>(
                    &candidate.key))
          return source && key->leg.ordinal == legOrdinal;
        const auto *key = std::get_if<loom::pnr::SystemTransferSinkTerminalKey>(
            &candidate.key);
        return !source && key && key->leg.ordinal == legOrdinal &&
               key->sinkOrdinal == 0;
      });
  require(found != service.transferTerminals.end(),
          "H omitted a canonical memory service terminal");
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
              "loom.system_pnr_search_domain.2.0",
          "H schema descriptor did not switch atomically to version 2.0");
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
  auto domain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, constraints, plan, {}, store));

  auto obligations = take(loom::mapping::projectSystemServiceObligations(
      dataflowView, constraints.view().rootThreadLaunches()));
  require(domain.serviceObligations().size() == obligations.size(),
          "H service-domain closure differs from the Dataflow projection");
  for (auto &&[service, obligation] :
       llvm::zip_equal(domain.serviceObligations(), obligations)) {
    require(service.key == obligation.key,
            "H service obligations are not in canonical key order");
    std::size_t expectedTerminals = 0;
    for (const auto &leg : obligation.legs) {
      const auto member = llvm::find(obligation.members, leg.member);
      require(member != obligation.members.end(),
              "projected leg names a foreign service member");
      expectedTerminals +=
          1 +
          (std::holds_alternative<dataflow::MessageTransferMemberRef>(*member)
               ? obligation.sinks.size()
               : 1);
    }
    require(service.transferTerminals.size() == expectedTerminals,
            "H omitted or added a canonical service terminal");
    if (std::holds_alternative<loom::mapping::TransferObligationFamilyKey>(
            service.key)) {
      require(!service.compatibleServiceRegions &&
                  !service.compatibleConsistencyDomains,
              "MessageTransfer acquired an operation-service target domain");
      require(
          llvm::all_of(service.transferTerminals,
                       [](const auto &terminal) {
                         return terminal.compatibleTransportEndpoints.empty();
                       }),
          "MessageTransfer reused a memory service-leg attachment");
      continue;
    }
    const auto &operation =
        std::get<loom::mapping::OperationServiceObligationFamilyKey>(
            service.key);
    if (std::holds_alternative<dataflow::FenceActorFamilyRef>(operation)) {
      require(!service.compatibleServiceRegions &&
                  service.compatibleConsistencyDomains,
              "fence target domain is not the typed consistency alternative");
    } else {
      require(
          service.compatibleServiceRegions &&
              !service.compatibleConsistencyDomains,
          "logical-memory target domain is not the typed region alternative");
    }
  }

  const auto &service = memoryServiceDomain(domain);
  require(service.compatibleServiceRegions &&
              !service.compatibleServiceRegions->empty(),
          "H omitted the builtin memory service region domain");
  require(service.transferTerminals.size() == 4,
          "one MemoryRead member must have source and sink for both legs");

  const auto requestCarriers = attachmentCarriers(system, 0, /*source=*/false);
  const auto responseCarriers = attachmentCarriers(system, 1, /*source=*/true);
  require(!requestCarriers.empty() && !responseCarriers.empty(),
          "builtin fixture has no service-leg carrier attachment");
  require(terminal(service, 0, false).compatibleTransportEndpoints ==
              requestCarriers,
          "MemoryRead request sink did not use the exact attachment union");
  require(terminal(service, 1, true).compatibleTransportEndpoints ==
              responseCarriers,
          "MemoryRead response source did not use the exact attachment union");
  require(terminal(service, 0, true).compatibleTransportEndpoints.empty() &&
              terminal(service, 1, false).compatibleTransportEndpoints.empty(),
          "H invented an unattached initiator endpoint for the builtin");

  const auto &fence = fenceServiceDomain(domain);
  require(
      fence.compatibleConsistencyDomains &&
          fence.compatibleConsistencyDomains->empty(),
      "builtin without fence capability did not expose exact infeasibility");
  require(fence.transferTerminals.size() == 4,
          "one MemoryFence member must have source and sink for both legs");

  constexpr llvm::StringLiteral oldDescriptor =
      "loom.system_pnr_search_domain.1.0";
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          {reinterpret_cast<const std::uint8_t *>(oldDescriptor.data()),
           oldDescriptor.size()},
          domain.canonicalViewBytes(), domain.digest(), store),
      "schema descriptor is not exact version 2.0",
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
  require(memoryServiceFraming.terminals.size() >= 2,
          "terminal strict-order fixture requires two terminals");
  auto duplicateTerminalBytes = duplicateFramedEntry(
      domain.canonicalViewBytes(), memoryServiceFraming.terminalCountOffset,
      memoryServiceFraming.terminalCount,
      memoryServiceFraming.terminals.front());
  auto duplicateTerminalDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateTerminalBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateTerminalBytes, duplicateTerminalDigest, store),
      "transfer-terminal domains are not strictly ordered",
      "strict H adoption accepted a duplicate service-terminal key");

  auto unorderedTerminalBytes = swapAdjacentFramedEntries(
      domain.canonicalViewBytes(), memoryServiceFraming.terminals[0],
      memoryServiceFraming.terminals[1]);
  auto unorderedTerminalDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedTerminalBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedTerminalBytes, unorderedTerminalDigest, store),
      "transfer-terminal domains are not strictly ordered",
      "strict H adoption accepted out-of-order service-terminal keys");

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
                                              transformConstraints,
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
      messageView, transformSystem, messageConstraints, messagePlan, {},
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
    std::size_t sourceCount = 0;
    std::size_t sinkCount = 0;
    for (const auto &terminal : messageService.transferTerminals) {
      if (std::holds_alternative<loom::pnr::SystemTransferSourceTerminalKey>(
              terminal.key)) {
        ++sourceCount;
        require(directlyAdmitted
                    ? terminal.compatibleTransportEndpoints ==
                          expectedMessageSources
                    : terminal.compatibleTransportEndpoints.empty(),
                "MessageTransfer source endpoint domain is not exact");
      } else {
        ++sinkCount;
        require(directlyAdmitted
                    ? terminal.compatibleTransportEndpoints ==
                          expectedMessageSinks
                    : terminal.compatibleTransportEndpoints.empty(),
                "MessageTransfer sink endpoint domain is not exact");
      }
    }
    require(sourceCount == obligation->legs.size(),
            "MessageTransfer omitted or added a source terminal");
    require(sinkCount == obligation->legs.size() * obligation->sinks.size(),
            "MessageTransfer omitted or added a sink terminal");
  }
  require(directlyAdmittedObligations > 0,
          "message fixture did not exercise direct endpoint admission");

  auto fenceDesign = buildFenceFabric(store, design.roots().front());
  auto fenceSystem =
      take(loom::fabric::requireSystemRoot(fenceDesign.roots().front().view()));
  auto fenceConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, fenceSystem, roots, store));
  auto fencePlan = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, fenceConstraints.view().rootThreadLaunches()));
  auto fenceDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, fenceSystem, fenceConstraints, fencePlan, {}, store));
  const auto &positiveFence = fenceServiceDomain(fenceDomain);
  const auto expectedConsistencyDomains =
      matchingFenceConsistencyDomains(fenceSystem, dataflowView, positiveFence);
  require(
      expectedConsistencyDomains.size() == 2,
      "fence fixture did not distinguish compatible and incompatible effects");
  require(positiveFence.compatibleConsistencyDomains &&
              *positiveFence.compatibleConsistencyDomains ==
                  expectedConsistencyDomains,
          "H fence target domain does not equal the exact compatible domains");
  auto adoptedFence = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      fenceDomain.canonicalViewBytes(), fenceDomain.digest(), store));
  require(fenceServiceDomain(adoptedFence).compatibleConsistencyDomains ==
              positiveFence.compatibleConsistencyDomains,
          "strict H adoption changed the fence consistency-domain projection");

  const FramedServiceSection fenceFraming =
      locateServiceSection(fenceDomain.canonicalViewBytes());
  const std::size_t fenceServiceIndex = static_cast<std::size_t>(
      &positiveFence - fenceDomain.serviceObligations().data());
  const auto &fenceTargetFraming =
      fenceFraming.entries[fenceServiceIndex].targetDomain;
  require(fenceTargetFraming && fenceTargetFraming->entries.size() == 2,
          "fence target-domain framing differs from the typed H domain");
  auto duplicateConsistencyBytes = duplicateFramedEntry(
      fenceDomain.canonicalViewBytes(), fenceTargetFraming->countOffset,
      fenceTargetFraming->count, fenceTargetFraming->entries.front());
  auto duplicateConsistencyDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateConsistencyBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateConsistencyBytes, duplicateConsistencyDigest, store),
      "Fabric target domain is not strictly ordered",
      "strict H adoption accepted a duplicate consistency-domain ref");

  auto unorderedConsistencyBytes = swapAdjacentFramedEntries(
      fenceDomain.canonicalViewBytes(), fenceTargetFraming->entries[0],
      fenceTargetFraming->entries[1]);
  auto unorderedConsistencyDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedConsistencyBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedConsistencyBytes, unorderedConsistencyDigest, store),
      "Fabric target domain is not strictly ordered",
      "strict H adoption accepted out-of-order consistency-domain refs");

  llvm::outs() << "System PnR service-domain anchors passed\n";
  return EXIT_SUCCESS;
}
