// Builds one portable memory fixture whose subordinate endpoint dispatches
// through Range provider rows to either its manager endpoint or its local
// storage, exports the Module SystemVerilog, and emits a testbench around the
// memory submodule alone. The testbench drives the subordinate endpoint with
// an accepted 64-bit address lane whose byte-address product leaves the
// 64-bit byte-address domain. The wrapped low bits of that product fall
// inside the manager-target Range row, so a decoder that compares the wrapped
// value forwards the request to the manager endpoint; the exact arithmetic
// selects no provider, and the request is never issued. Control requests
// prove that both provider targets decode and that the decoder is not wedged
// after a rejected request.
//
// The fixture keeps every memory endpoint inside the Module (a subordinate
// only storage behind the manager, an engine memory in front of the
// subordinate), so the Module boundary carries transport ports only and the
// memory submodule exposes its manager and subordinate service ports to the
// testbench directly.

#include "ConfigurationABITestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/MemoryServiceTransport.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t kLocalCapacityBytes = 4096;
constexpr std::uint32_t kElementWidthBits = 32;
constexpr std::uint32_t kLaneWidthBits = 64;
/// Byte addresses below this boundary dispatch to the manager endpoint; the
/// rest of the local capacity dispatches to the local storage.
constexpr std::uint64_t kManagerRangeBytes = 16;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test.str() << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-memory-service-address-arithmetic-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

::fabric::UnsignedDomain singleton(llvm::StringRef test, std::uint64_t value) {
  return take(test, ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

::fabric::ResourceContract singleUseResourceContract(llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return take(test, ::fabric::ResourceContract::create(std::move(declaration)));
}

/// One element access class with the widest lane the portable profile
/// admits: a 64-bit root-relative index of 32-bit elements, for a load or a
/// store capability.
::fabric::MemoryAccessClass wideLaneElementAccess(llvm::StringRef test,
                                                  bool store) {
  auto alignment = take(
      test, ::fabric::AlignmentDomain::create(take(
                test, ::fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      test,
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {store ? ::fabric::ReadSubwordSemantics::NotApplicable
                 : ::fabric::ReadSubwordSemantics::Exact}));
  auto write = take(
      test,
      ::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::fromCanonical(
          {store ? ::fabric::WriteSubwordSemantics::Exact
                 : ::fabric::WriteSubwordSemantics::NotApplicable}));
  return take(test,
              ::fabric::MemoryAccessClass::create(
                  ::dataflow::semantics::MemoryAccessForm::Element,
                  singleton(test, kElementWidthBits), singleton(test, 1),
                  {{::dataflow::semantics::MemoryMaskForm::Absent,
                    ::fabric::InactiveLaneSemantics::NotApplicable}},
                  std::move(alignment), std::move(read), std::move(write),
                  take(test, ::fabric::MemoryAddressDomain::rootRelative(
                                 singleton(test, kLaneWidthBits)))));
}

::fabric::MemoryServiceCapabilityDeclaration
plainCapability(llvm::StringRef test, ::dataflow::OperationSchemaId schema) {
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  return {take(test, ::fabric::MemoryActorContractDomain::create(schema,
                                                                 {plain})),
          take(test, ::fabric::ParameterizedMemoryAccessDomain::create(
                         {wideLaneElementAccess(
                             test, schema == ::dataflow::OperationSchemaId::
                                                 DataflowStore)})),
          {0},
          128,
          {::fabric::UsePatternKey(0)},
          ::fabric::NoMemoryServiceConsistency{}};
}

loom::adg::LocalMemoryServiceSpec
localStorageService(llvm::StringRef test, mlir::MLIRContext &contractContext) {
  ::fabric::MemoryServiceContractDeclaration declaration{
      {{0, kLocalCapacityBytes, ::fabric::MemoryServiceRegionBehavior::Storage,
        std::nullopt}},
      singleUseResourceContract(test),
      {plainCapability(test, ::dataflow::OperationSchemaId::DataflowLoad),
       plainCapability(test, ::dataflow::OperationSchemaId::DataflowStore)}};
  auto contract = take(test, ::fabric::MemoryServiceContractRecord::create(
                                 &contractContext,
                                 ::fabric::MemoryServiceOwnerKind::Local,
                                 std::move(declaration)));
  return take(test, loom::adg::LocalMemoryServiceSpec::create(
                        kLocalCapacityBytes, contract));
}

/// One plain load operation port over the same wide-lane access class. A
/// manager endpoint requires an operation engine; the engines are left
/// unconfigured, because the test drives only the subordinate endpoint.
::fabric::MemoryOperationPortDeclaration loadPortDeclaration(
    llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration resource;
  resource.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  resource.requesters = {::fabric::RequesterKey(0)};
  resource.eligibilityCount = 1;
  resource.eventCount = 1;
  resource.timingContracts = {{::fabric::TimingContractKey(0), {0}}};
  resource.usePatterns = {{::fabric::UsePatternKey(0),
                           ::fabric::RequesterKey(0),
                           ::fabric::EligibilityKey(0),
                           ::fabric::EventKey(0),
                           ::fabric::EventKey(0),
                           std::nullopt,
                           ::fabric::TimingContractKey(0),
                           {},
                           {{{}}}}};
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  return {{0, 1, 2, 3},
          take(test, ::fabric::ResourceContract::create(std::move(resource))),
          {{::fabric::MemoryPortTransactionProjection::Direct}},
          {{take(test, ::fabric::MemoryActorContractDomain::create(
                           ::dataflow::OperationSchemaId::DataflowLoad,
                           {plain})),
            {{::dataflow::semantics::ServiceValueRole::Address, 0},
             {::dataflow::semantics::ServiceValueRole::Data, 2},
             {::dataflow::semantics::ServiceValueRole::Control, 1},
             {::dataflow::semantics::ServiceValueRole::Completion, 3}},
            take(test, ::fabric::ParameterizedMemoryAccessDomain::create(
                           {wideLaneElementAccess(test, false)})),
            {::fabric::UsePatternKey(0)}}}};
}

::fabric::MemoryDispatchTarget localTarget() {
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
}

::fabric::MemoryDispatchTarget managerTarget(std::uint64_t ordinal) {
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{ordinal});
}

struct Fixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
  loom::hardware::FinalizedConfigurationABI abi;
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  /// The memory under test: the one occurrence owning both a manager and a
  /// subordinate endpoint.
  loom::fabric::FabricMemoryOccurrenceRef memory;
};

/// Three memories close every memory endpoint inside the Module: the
/// subordinate-only storage behind the manager of the memory under test, and
/// the engine memory whose manager drives the subordinate under test.
loom::fabric::FinalizedFabricRoot
makeMemoryModule(llvm::StringRef test, const loom::ArtifactStore &store,
                 mlir::MLIRContext &contractContext) {
  using namespace loom::adg;
  const PortType byte = take(test, PortType::bits(8));
  const PortType memoryPort =
      take(test, PortType::memory({PortType::kDynamicExtent}, byte));
  const PortType address = take(test, PortType::bits(kLaneWidthBits));
  const PortType data = take(test, PortType::bits(kElementWidthBits));
  const PortType control = take(test, PortType::bits(0));

  ::fabric::MemoryConnectivityDeclaration storageConnectivity;
  ::fabric::MemorySubordinateDispatchDeclaration storageSubordinate;
  storageSubordinate.maxExposedBindings = 1;
  storageSubordinate.targetDomain = {localTarget()};
  storageConnectivity.subordinateEndpoints.push_back(
      std::move(storageSubordinate));
  auto storageSpec = take(
      test, MemorySpec::create(
                {}, {memoryPort}, {}, {0}, std::nullopt,
                localStorageService(test, contractContext),
                take(test, MemoryConnectivitySpec::create(
                               std::move(storageConnectivity)))));

  ::fabric::MemoryConnectivityDeclaration testedConnectivity;
  ::fabric::MemoryOperationPortDispatchDeclaration testedOperationPort;
  testedOperationPort.capabilityTargetDomains = {{localTarget()}};
  testedConnectivity.operationPorts.push_back(std::move(testedOperationPort));
  ::fabric::MemorySubordinateDispatchDeclaration testedSubordinate;
  testedSubordinate.maxExposedBindings = 2;
  testedSubordinate.matchFields = {::fabric::MemoryProviderMatchField::Range};
  testedSubordinate.targetDomain = {managerTarget(0), localTarget()};
  testedConnectivity.subordinateEndpoints.push_back(
      std::move(testedSubordinate));
  const std::vector<PortType> testedInputs{memoryPort, address, control};
  const std::vector<PortType> testedOutputs{data, control, memoryPort};
  auto testedSpec = take(
      test, MemorySpec::create(
                testedInputs, testedOutputs, {0}, {2},
                MemoryEngineSpec::spatial({loadPortDeclaration(test)}),
                localStorageService(test, contractContext),
                take(test, MemoryConnectivitySpec::create(
                               std::move(testedConnectivity)))));

  ::fabric::MemoryConnectivityDeclaration driverConnectivity;
  ::fabric::MemoryOperationPortDispatchDeclaration driverOperationPort;
  driverOperationPort.capabilityTargetDomains = {{managerTarget(0)}};
  driverConnectivity.operationPorts.push_back(std::move(driverOperationPort));
  auto driverSpec = take(
      test, MemorySpec::create(
                testedInputs, {data, control}, {0}, {},
                MemoryEngineSpec::spatial({loadPortDeclaration(test)}),
                std::nullopt,
                take(test, MemoryConnectivitySpec::create(
                               std::move(driverConnectivity)))));

  DesignBuilder design(store);
  const std::vector<PortType> moduleInputs{address, control, address, control};
  const std::vector<PortType> moduleOutputs{data, control, data, control};
  auto core = take(test, design.createSpatialCore("memory-address-arithmetic",
                                                  moduleInputs,
                                                  moduleOutputs));
  auto storage = take(test, core.addMemory({}, storageSpec));
  require(test, storage.values().size() == 1,
          "storage memory exposes one subordinate output");
  auto tested = take(test, core.addMemory({storage.values().front(),
                                           take(test, core.input(0)),
                                           take(test, core.input(1))},
                                          testedSpec));
  require(test, tested.values().size() == 3,
          "tested memory exposes its results and the subordinate");
  auto driver = take(test, core.addMemory({tested.values()[2],
                                           take(test, core.input(2)),
                                           take(test, core.input(3))},
                                          driverSpec));
  require(test, driver.values().size() == 2,
          "driver memory exposes its engine results");
  requireSuccess(test, core.close({tested.values()[0], tested.values()[1],
                                   driver.values()[0], driver.values()[1]}));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "memory fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

Fixture makeFixture(llvm::StringRef test, const loom::ArtifactStore &store,
                    mlir::MLIRContext &contractContext) {
  loom::fabric::FinalizedFabricRoot module =
      makeMemoryModule(test, store, contractContext);
  std::optional<loom::fabric::FabricMemoryOccurrenceRef> tested;
  for (loom::fabric::FabricMemoryOccurrenceRef memory :
       module.view().memoryOccurrences()) {
    const auto *connectivity = module.view().memoryConnectivity(memory);
    require(test, connectivity != nullptr,
            "memory occurrence has no connectivity contract");
    if (connectivity->subordinateEndpoints().size() == 1 &&
        !connectivity->operationPorts().empty()) {
      require(test, !tested.has_value(),
              "memory fixture has one memory under test");
      tested = memory;
    }
  }
  require(test, tested.has_value(), "memory fixture lost the memory under test");
  loom::fabric::FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSpatialCoreSystem(module, store, 1));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore =
      take(test, loom::hardware::test::requireSingleSpatialCoreOccurrence(
                     system));
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  for (loom::fabric::FabricMemoryOccurrenceRef memory :
       module.view().memoryOccurrences()) {
    auto schema = take(test, module.view().memoryConfigurationSchema(memory));
    auto target = take(test, loom::fabric::FabricModulePhysicalTargetRef::create(
                                 schema.field()));
    auto physical =
        take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                       loom::fabric::SpatialCoreInternalOccurrenceRef{
                           spatialCore, std::move(target)}));
    overrides.push_back(
        {physical,
         loom::hardware::DirectBitsEncoding{schema.layout().carrierBitCount},
         std::vector<std::uint8_t>((schema.layout().carrierBitCount + 7) / 8,
                                   0)});
  }
  auto draft = take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                              system, overrides));
  loom::hardware::FinalizedConfigurationABI abi = take(
      test, loom::hardware::finalizeConfigurationABI(std::move(draft), store));
  return Fixture{std::move(module), std::move(system), std::move(abi),
                 spatialCore, *tested};
}

/// The active configuration of the memory under test with its two provider
/// rows: bytes below the manager boundary forward to the manager endpoint,
/// the rest of the local capacity is served locally. The bytes are the direct
/// carrier the memory submodule decodes, least significant byte first.
loom::CanonicalSemanticBytes
makeActiveConfiguration(llvm::StringRef test, const Fixture &fixture) {
  auto schema = take(test, fixture.module.view().memoryConfigurationSchema(
                               fixture.memory));
  loom::fabric::FabricMemoryActive active;
  active.operationRows.resize(schema.layout().operationRows.size());
  active.providerDecodeRows.resize(schema.layout().providerRows.size());
  require(test, active.providerDecodeRows.size() == 1,
          "memory under test exposes one subordinate endpoint");
  require(test, schema.layout().providerRows.front().size() == 2,
          "memory under test exposes two provider rows");
  active.providerDecodeRows.front().resize(2);
  active.providerDecodeRows.front()[0] =
      loom::fabric::FabricMemoryProviderDecodeRow{
          {loom::fabric::FabricMemoryRangeMatch{0, kManagerRangeBytes}},
          managerTarget(0),
          0};
  active.providerDecodeRows.front()[1] =
      loom::fabric::FabricMemoryProviderDecodeRow{
          {loom::fabric::FabricMemoryRangeMatch{
              kManagerRangeBytes, kLocalCapacityBytes - kManagerRangeBytes}},
          localTarget(),
          0};
  return take(test, schema.encode(loom::fabric::FabricMemoryConfigurationValue{
                        std::move(active)}));
}

/// One exported port of the memory submodule: its name, direction, and width.
struct ModulePort final {
  std::string name;
  bool output = false;
  std::uint64_t width = 0;
};

/// The memory submodule under test and its ports, in declaration order.
struct MemorySubmodule final {
  std::string name;
  std::vector<ModulePort> ports;
  std::string managerPrefix;
  std::string subordinatePrefix;
  std::uint64_t configurationWidthBits = 0;
};

MemorySubmodule findMemorySubmodule(llvm::StringRef test, mlir::ModuleOp module,
                                    loom::fabric::FabricMemoryOccurrenceRef memory) {
  const std::string expected = "loom_memory_" + std::to_string(memory.id());
  circt::hw::HWModuleOp found;
  module.walk([&](circt::hw::HWModuleOp candidate) {
    if (candidate.getSymName() == expected)
      found = candidate;
  });
  require(test, static_cast<bool>(found), "fixture omitted the memory under test");
  MemorySubmodule submodule;
  submodule.name = expected;
  const std::regex servicePattern("^(service_[0-9]+)_request_valid$");
  for (const circt::hw::PortInfo &port : found.getPortList()) {
    const std::int64_t width = mlir::isa<circt::seq::ClockType>(port.type)
                                   ? 1
                                   : circt::hw::getBitWidth(port.type);
    // Zero-width payloads are omitted from the exported SystemVerilog.
    if (width <= 0)
      continue;
    submodule.ports.push_back({port.getName().str(), port.isOutput(),
                               static_cast<std::uint64_t>(width)});
    std::smatch match;
    const std::string name = port.getName().str();
    if (std::regex_match(name, match, servicePattern)) {
      std::string &prefix =
          port.isOutput() ? submodule.managerPrefix : submodule.subordinatePrefix;
      require(test, prefix.empty(),
              "memory under test exposes one endpoint per role");
      prefix = match[1];
    } else if (name == "configuration_value") {
      submodule.configurationWidthBits = static_cast<std::uint64_t>(width);
    }
  }
  require(test,
          !submodule.managerPrefix.empty() &&
              !submodule.subordinatePrefix.empty() &&
              submodule.configurationWidthBits != 0,
          "memory under test lost a service endpoint or its configuration");
  return submodule;
}

std::string hexLiteral(std::uint64_t widthBits,
                       llvm::ArrayRef<std::uint8_t> bytesLittleEndian) {
  std::string digits;
  llvm::raw_string_ostream out(digits);
  for (std::uint8_t byte : llvm::reverse(bytesLittleEndian))
    out << llvm::format_hex_no_prefix(byte, 2, true);
  return std::to_string(widthBits) + "'h" + digits;
}

/// Renders the conformance testbench around the memory submodule. The
/// subordinate service port receives the test's requests; the manager
/// service port is answered by the testbench acting as the external memory
/// service. Every other submodule input idles at zero.
std::string renderTestbench(const MemorySubmodule &submodule,
                            const loom::CanonicalSemanticBytes &configuration) {
  std::ostringstream out;
  out << "module memory_address_testbench;\n";
  for (const ModulePort &port : submodule.ports) {
    out << "  logic ";
    if (port.width != 1)
      out << '[' << (port.width - 1) << ":0] ";
    out << port.name << ";\n";
  }
  out << "\n  " << submodule.name << " dut(.*);\n"
      << "  always #5 clock = ~clock;\n";
  std::ostringstream idle;
  for (const ModulePort &port : submodule.ports) {
    if (port.output || port.name == "clock" || port.name == "reset" ||
        port.name == "configuration_value" ||
        llvm::StringRef(port.name).starts_with(submodule.managerPrefix) ||
        llvm::StringRef(port.name).starts_with(submodule.subordinatePrefix))
      continue;
    idle << "    " << port.name << " = '0;\n";
  }
  const std::string idleInputs = idle.str();
  const std::string manager = submodule.managerPrefix;
  const std::string subordinate = submodule.subordinatePrefix;
  out << R"sv(
  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  initial begin
    repeat (4096) @(posedge clock);
    $fatal(1, "RTL testbench cycle limit exceeded");
  end

  // The external memory service behind the manager endpoint: every accepted
  // request is answered one cycle later with a fixed pattern, and every
  // forwarded request is counted with its lane address.
  integer manager_request_count;
  logic [63:0] manager_last_address;
  logic [63:0] manager_last_base;
  always @(posedge clock or posedge reset) begin
    if (reset) begin
      manager_request_count <= 0;
      manager_last_address <= 0;
      manager_last_base <= 0;
      MANAGER_response_valid <= 0;
    end else begin
      if (MANAGER_response_valid && MANAGER_response_ready)
        MANAGER_response_valid <= 0;
      if (MANAGER_request_valid && MANAGER_request_ready) begin
        manager_request_count <= manager_request_count + 1;
        manager_last_address <= MANAGER_request_address;
        manager_last_base <= MANAGER_request_base_address;
        MANAGER_response_valid <= 1;
      end
    end
  end

  integer subordinate_response_count;
  logic [31:0] subordinate_last_data;
  always @(posedge clock or posedge reset) begin
    if (reset) begin
      subordinate_response_count <= 0;
      subordinate_last_data <= 0;
    end else if (SUBORDINATE_response_valid && SUBORDINATE_response_ready) begin
      subordinate_response_count <= subordinate_response_count + 1;
      subordinate_last_data <= SUBORDINATE_response_data;
    end
  end

  // Presents one subordinate request and reports whether the memory issued
  // it within the wait budget. A request the decoder rejects is withdrawn
  // unissued.
  task automatic present_request(
      input logic kind, input logic [63:0] address, input logic [63:0] base,
      input logic [31:0] lane_width, input logic [31:0] data,
      input integer wait_cycles, output bit accepted);
    integer cycles;
    begin
      @(negedge clock);
      SUBORDINATE_request_kind = kind;
      SUBORDINATE_request_address = address;
      SUBORDINATE_request_data = data;
      SUBORDINATE_request_mask = '0;
      SUBORDINATE_request_active_lanes_kind = 0;
      SUBORDINATE_request_access_form = 2'd0;
      SUBORDINATE_request_address_form = 1'b0;
      SUBORDINATE_request_element_width = 64'd)sv"
      << kElementWidthBits << R"sv(;
      SUBORDINATE_request_lane_count = 64'd1;
      SUBORDINATE_request_address_lane_width = lane_width;
      SUBORDINATE_request_base_address = base;
      SUBORDINATE_request_context = 64'd0;
      SUBORDINATE_request_valid = 1;
      accepted = 0;
      cycles = 0;
      while (!accepted && cycles != wait_cycles) begin
        #1;
        accepted = SUBORDINATE_request_ready;
        @(posedge clock);
        @(negedge clock);
        cycles = cycles + 1;
      end
      SUBORDINATE_request_valid = 0;
    end
  endtask

  task automatic wait_response(input integer expected_count,
                               input string message);
    integer cycles;
    begin
      cycles = 0;
      while (subordinate_response_count != expected_count && cycles != 32) begin
        @(negedge clock);
        cycles = cycles + 1;
      end
      check(subordinate_response_count == expected_count, message);
    end
  endtask

  bit accepted;
  initial begin
    clock = 0;
    reset = 1;
)sv" << idleInputs
      << R"sv(    configuration_value = )sv"
      << hexLiteral(submodule.configurationWidthBits, configuration.bytes())
      << R"sv(;
    MANAGER_request_ready = 1;
    MANAGER_response_data = 32'h12345678;
    SUBORDINATE_request_valid = 0;
    SUBORDINATE_response_ready = 1;
    SUBORDINATE_request_kind = 0;
    SUBORDINATE_request_address = 0;
    SUBORDINATE_request_data = 0;
    SUBORDINATE_request_mask = '0;
    SUBORDINATE_request_active_lanes_kind = 0;
    SUBORDINATE_request_access_form = 0;
    SUBORDINATE_request_address_form = 0;
    SUBORDINATE_request_element_width = 0;
    SUBORDINATE_request_lane_count = 0;
    SUBORDINATE_request_address_lane_width = 0;
    SUBORDINATE_request_base_address = 0;
    SUBORDINATE_request_context = 0;
    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    repeat (2) @(posedge clock);

    // Control: the local target. A store at lane 4 (byte 16, the first
    // locally served byte) reads back through the same decoder.
    present_request(1'b1, 64'd4, 64'd0, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'hcafebabe, 20, accepted);
    check(accepted, "local store was not issued");
    wait_response(1, "local store did not complete");
    present_request(1'b0, 64'd4, 64'd0, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'd0, 20, accepted);
    check(accepted, "local load was not issued");
    wait_response(2, "local load did not complete");
    check(subordinate_last_data == 32'hcafebabe,
          "local load returned the wrong data");
    check(manager_request_count == 0,
          "locally served requests reached the manager endpoint");

    // Control: the manager target. Lane 1 is byte 4, inside the manager row.
    present_request(1'b0, 64'd1, 64'd0, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'd0, 20, accepted);
    check(accepted, "manager load was not issued");
    wait_response(3, "manager load did not complete");
    check(manager_request_count == 1 && manager_last_address == 64'd1 &&
              manager_last_base == 64'd0,
          "manager load was not forwarded with its lane address");
    check(subordinate_last_data == 32'h12345678,
          "manager load returned the wrong data");

    // An accepted 64-bit lane whose product wraps: lane 2^62 times four
    // bytes is 2^64, whose wrapped low bits (zero) lie inside the manager
    // row. The exact address is outside the byte-address domain, so no
    // provider row matches and the request is never issued or forwarded.
    present_request(1'b0, 64'h4000000000000000, 64'd0, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'd0, 32, accepted);
    check(!accepted, "wrapping lane product selected a provider");
    check(manager_request_count == 1 && subordinate_response_count == 3,
          "wrapping lane product was forwarded or answered");

    // A base address whose sum with the lane product wraps to zero.
    present_request(1'b0, 64'd1, 64'hfffffffffffffffc, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'd0, 32, accepted);
    check(!accepted, "wrapping base sum selected a provider");
    check(manager_request_count == 1 && subordinate_response_count == 3,
          "wrapping base sum was forwarded or answered");

    // A lane width beyond the byte-address domain has no exact address.
    present_request(1'b0, 64'd1, 64'd0, 32'd)sv"
      << (kLaneWidthBits + 1) << R"sv(, 32'd0, 32, accepted);
    check(!accepted, "unsupported lane width selected a provider");
    check(manager_request_count == 1 && subordinate_response_count == 3,
          "unsupported lane width was forwarded or answered");

    // The decoder is not wedged: the manager control still decodes.
    present_request(1'b0, 64'd2, 64'd0, 32'd)sv"
      << kLaneWidthBits << R"sv(, 32'd0, 20, accepted);
    check(accepted, "manager load after rejected requests was not issued");
    wait_response(4, "manager load after rejected requests did not complete");
    check(manager_request_count == 2 && manager_last_address == 64'd2,
          "manager load after rejected requests was not forwarded");

    $write("memory_address_arithmetic_passed\n");
    $finish;
  end
endmodule
)sv";
  // The body names the two service ports by role placeholders; bind them to
  // the exported prefixes.
  std::string rendered = out.str();
  for (const auto &[placeholder, prefix] :
       {std::make_pair(std::string("MANAGER"), manager),
        std::make_pair(std::string("SUBORDINATE"), subordinate)}) {
    std::size_t position = 0;
    while ((position = rendered.find(placeholder, position)) !=
           std::string::npos) {
      rendered.replace(position, placeholder.size(), prefix);
      position += prefix.size();
    }
  }
  return rendered;
}

void writeArtifacts(const std::filesystem::path &root,
                    llvm::StringRef systemVerilog, llvm::StringRef testbench) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "memory_address_module.sv") << systemVerilog.str();
  std::ofstream(root / "memory_address_testbench.sv") << testbench.str();
}

} // namespace

int main(int argc, char **argv) {
  const llvm::StringRef test = "memory_service_address_arithmetic";
  if (argc != 1 && argc != 2)
    fail(test, "expected at most one output directory");
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext contractContext(mlir::MLIRContext::Threading::DISABLED);
  const Fixture fixture = makeFixture(test, store, contractContext);

  const auto layout = take(
      test, loom::hardware::rtl::derivePortableMemoryServiceLayout(
                fixture.module.view()));
  require(test, layout.maximumAddressLaneWidthBits == kLaneWidthBits,
          "memory fixture does not carry a 64-bit address lane");
  const auto arithmetic =
      loom::hardware::rtl::derivePortableMemoryAddressArithmetic(layout);
  // A 64-bit lane times an element byte count below 2^61 plus two 64-bit
  // terms is exact in twice the byte-address width.
  require(test,
          arithmetic && arithmetic->laneWidthBits == kLaneWidthBits &&
              arithmetic->calculationWidthBits >=
                  arithmetic->byteAddressWidthBits + kLaneWidthBits,
          "portable address arithmetic does not admit the 64-bit lane in a "
          "wider calculation domain");
  loom::hardware::rtl::PortableMemoryServiceLayout tooWide = layout;
  tooWide.maximumAddressLaneWidthBits = kLaneWidthBits + 1;
  require(test,
          !loom::hardware::rtl::derivePortableMemoryAddressArithmetic(tooWide),
          "a lane beyond the byte-address domain was admitted");

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  // A verification failure names the offending operation, not only its
  // message.
  mlir::ScopedDiagnosticHandler diagnostics(
      &context, [](mlir::Diagnostic &diagnostic) {
        diagnostic.print(llvm::errs());
        llvm::errs() << '\n';
        for (mlir::Diagnostic &note : diagnostic.getNotes()) {
          note.print(llvm::errs());
          llvm::errs() << '\n';
        }
        return mlir::success();
      });
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, fixture.spatialCore, fixture.abi));
  const MemorySubmodule submodule =
      findMemorySubmodule(test, *skeleton.module, fixture.memory);
  const std::string systemVerilog = take(
      test, loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
                *skeleton.module));
  require(test,
          llvm::StringRef(systemVerilog).contains("module " + submodule.name),
          "memory fixture RTL lost the memory under test");
  llvm::outs() << "memory_address_arithmetic lane_bits="
               << arithmetic->laneWidthBits << " byte_address_bits="
               << arithmetic->byteAddressWidthBits << " calculation_bits="
               << arithmetic->calculationWidthBits << " submodule="
               << submodule.name << '\n';

  if (argc == 2) {
    const loom::CanonicalSemanticBytes configuration =
        makeActiveConfiguration(test, fixture);
    require(test,
            configuration.bytes().size() * 8 >= submodule.configurationWidthBits,
            "memory configuration carrier is narrower than the submodule port");
    writeArtifacts(argv[1], systemVerilog,
                   renderTestbench(submodule, configuration));
  }
  return EXIT_SUCCESS;
}
