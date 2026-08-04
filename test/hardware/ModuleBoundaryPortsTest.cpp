#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Hardware/RTL/Transport.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <array>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using loom::fabric::FabricModuleBoundaryEndpointRef;
using loom::fabric::FabricPortDirection;
using loom::hardware::rtl::ModuleBoundaryTransportPortProjection;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-module-boundary-ports-test", path))
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

unsigned width(const circt::hw::PortInfo &port) {
  return mlir::cast<mlir::IntegerType>(port.type).getWidth();
}

void requirePort(llvm::StringRef test,
                 const std::optional<circt::hw::PortInfo> &port,
                 unsigned expectedWidth,
                 circt::hw::ModulePort::Direction expectedDirection,
                 llvm::StringRef description) {
  require(test, port.has_value(), (description + " is absent").str());
  require(test, width(*port) == expectedWidth,
          (description + " has the wrong width").str());
  require(test, port->dir == expectedDirection,
          (description + " has the wrong direction").str());
}

void requireNoPort(llvm::StringRef test,
                   const std::optional<circt::hw::PortInfo> &port,
                   llvm::StringRef description) {
  require(test, !port, (description + " is unexpectedly present").str());
}

void requireControlPorts(
    llvm::StringRef test,
    const ModuleBoundaryTransportPortProjection &projection,
    circt::hw::ModulePort::Direction forwardDirection,
    circt::hw::ModulePort::Direction readyDirection) {
  require(test,
          width(projection.valid) == 1 &&
              projection.valid.dir == forwardDirection,
          "valid port changed width or direction");
  require(test,
          width(projection.ready) == 1 &&
              projection.ready.dir == readyDirection,
          "ready port changed width or direction");
}

void requireSamePort(llvm::StringRef test, const circt::hw::PortInfo &left,
                     const circt::hw::PortInfo &right) {
  require(test,
          left.name == right.name && left.type == right.type &&
              left.dir == right.dir,
          "repeated projection changed a CIRCT port");
}

void requireSameOptionalPort(llvm::StringRef test,
                             const std::optional<circt::hw::PortInfo> &left,
                             const std::optional<circt::hw::PortInfo> &right) {
  require(test, left.has_value() == right.has_value(),
          "repeated projection changed optional port presence");
  if (left)
    requireSamePort(test, *left, *right);
}

void finalizedModuleProjectsExactTransportBoundary() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());

  const loom::adg::PortType bits8 = take(test, loom::adg::PortType::bits(8));
  const loom::adg::PortType bits0 = take(test, loom::adg::PortType::bits(0));
  const loom::adg::PortType tagged16x4 =
      take(test, loom::adg::PortType::taggedBits(16, 4));
  const loom::adg::PortType tagged0x3 =
      take(test, loom::adg::PortType::taggedBits(0, 3));
  const loom::adg::PortType memory8 =
      take(test, loom::adg::PortType::memory(
                     {loom::adg::PortType::kDynamicExtent}, bits8));

  loom::adg::DesignBuilder design(store);
  auto module = take(test, design.createSpatialCore(
                               "module-boundary-ports",
                               {bits8, memory8, bits0, tagged16x4, tagged0x3},
                               {tagged0x3, bits8, tagged16x4, bits0}));
  std::vector<loom::adg::SpatialValue> inputs;
  for (std::size_t ordinal = 0; ordinal != 5; ++ordinal)
    inputs.push_back(take(test, module.input(ordinal)));
  if (llvm::Error error =
          module.close({inputs[4], inputs[0], inputs[3], inputs[2]}))
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "fixture did not publish exactly one Module root");
  const loom::fabric::FabricArtifactView &view =
      finalized.roots().front().view();

  const auto moduleTemplate = view.moduleRootTemplate();
  require(test, moduleTemplate.has_value(),
          "Module root has no unique template reference");
  require(test,
          view.moduleBoundaryEndpointCount(*moduleTemplate,
                                           FabricPortDirection::Input) == 5 &&
              view.moduleBoundaryEndpointCount(
                  *moduleTemplate, FabricPortDirection::Output) == 4,
          "Module root changed its exact signature cardinality");

  const FabricModuleBoundaryEndpointRef memoryBoundary{
      *moduleTemplate, FabricPortDirection::Input, 1};
  require(test, !view.moduleBoundaryEndpointDataPath(memoryBoundary),
          "memory boundary acquired a token data path");
  const FabricModuleBoundaryEndpointRef taggedBoundary{
      *moduleTemplate, FabricPortDirection::Input, 3};
  const auto taggedType = view.moduleBoundaryEndpointDataPath(taggedBoundary);
  require(test,
          taggedType && taggedType->kind == ::fabric::DataPathKind::BitsTag &&
              taggedType->payloadWidthBits == 16 &&
              taggedType->tagWidthBits == 4,
          "tagged boundary data path changed during strict import");

  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  std::vector<ModuleBoundaryTransportPortProjection> ports = take(
      test,
      loom::hardware::rtl::deriveModuleBoundaryTransportPorts(builder, view));
  std::vector<ModuleBoundaryTransportPortProjection> repeated = take(
      test,
      loom::hardware::rtl::deriveModuleBoundaryTransportPorts(builder, view));
  require(test, ports.size() == 8,
          "memory boundary was projected as a token port");
  require(test, repeated.size() == ports.size(),
          "repeated projection changed port cardinality");

  std::set<std::string> names;
  for (std::size_t ordinal = 0; ordinal != ports.size(); ++ordinal) {
    const auto insert = [&](const circt::hw::PortInfo &port) {
      require(test, !port.getName().empty(), "projected an unnamed port");
      require(test, names.insert(port.getName().str()).second,
              "projected duplicate CIRCT port names");
    };
    if (ports[ordinal].data)
      insert(*ports[ordinal].data);
    if (ports[ordinal].tag)
      insert(*ports[ordinal].tag);
    insert(ports[ordinal].valid);
    insert(ports[ordinal].ready);
    require(test, ports[ordinal].boundary == repeated[ordinal].boundary,
            "repeated projection changed boundary order");
    requireSameOptionalPort(test, ports[ordinal].data, repeated[ordinal].data);
    requireSameOptionalPort(test, ports[ordinal].tag, repeated[ordinal].tag);
    requireSamePort(test, ports[ordinal].valid, repeated[ordinal].valid);
    requireSamePort(test, ports[ordinal].ready, repeated[ordinal].ready);
  }

  const std::array<FabricModuleBoundaryEndpointRef, 8> expectedReferences{{
      {*moduleTemplate, FabricPortDirection::Input, 0},
      {*moduleTemplate, FabricPortDirection::Input, 2},
      {*moduleTemplate, FabricPortDirection::Input, 3},
      {*moduleTemplate, FabricPortDirection::Input, 4},
      {*moduleTemplate, FabricPortDirection::Output, 0},
      {*moduleTemplate, FabricPortDirection::Output, 1},
      {*moduleTemplate, FabricPortDirection::Output, 2},
      {*moduleTemplate, FabricPortDirection::Output, 3},
  }};
  for (std::size_t ordinal = 0; ordinal != ports.size(); ++ordinal)
    require(test, ports[ordinal].boundary == expectedReferences[ordinal],
            "transport projection changed direction/ordinal order");

  using Direction = circt::hw::ModulePort::Direction;
  requirePort(test, ports[0].data, 8, Direction::Input, "input bits data");
  requireNoPort(test, ports[0].tag, "input bits tag");
  requireControlPorts(test, ports[0], Direction::Input, Direction::Output);

  requireNoPort(test, ports[1].data, "input zero-width bits data");
  requireNoPort(test, ports[1].tag, "input zero-width bits tag");
  requireControlPorts(test, ports[1], Direction::Input, Direction::Output);

  requirePort(test, ports[2].data, 16, Direction::Input, "input tagged data");
  requirePort(test, ports[2].tag, 4, Direction::Input, "input tagged tag");
  requireControlPorts(test, ports[2], Direction::Input, Direction::Output);

  requireNoPort(test, ports[3].data, "input zero-width tagged data");
  requirePort(test, ports[3].tag, 3, Direction::Input,
              "input zero-width tagged tag");
  requireControlPorts(test, ports[3], Direction::Input, Direction::Output);

  requireNoPort(test, ports[4].data, "output zero-width tagged data");
  requirePort(test, ports[4].tag, 3, Direction::Output,
              "output zero-width tagged tag");
  requireControlPorts(test, ports[4], Direction::Output, Direction::Input);

  requirePort(test, ports[5].data, 8, Direction::Output, "output bits data");
  requireNoPort(test, ports[5].tag, "output bits tag");
  requireControlPorts(test, ports[5], Direction::Output, Direction::Input);

  requirePort(test, ports[6].data, 16, Direction::Output, "output tagged data");
  requirePort(test, ports[6].tag, 4, Direction::Output, "output tagged tag");
  requireControlPorts(test, ports[6], Direction::Output, Direction::Input);

  requireNoPort(test, ports[7].data, "output zero-width bits data");
  requireNoPort(test, ports[7].tag, "output zero-width bits tag");
  requireControlPorts(test, ports[7], Direction::Output, Direction::Input);
}

void nonModuleAndUnrepresentableWidthsFailWithoutIr() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());

  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> top =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(top->getBody());

  loom::adg::FinalizedFabricDesign builtin =
      take(test, loom::adg::buildBuiltinTarget(
                     store, loom::adg::BuiltinTargetPreset::Small));
  require(test, !builtin.roots().front().view().moduleRootTemplate(),
          "System root acquired a Module template reference");
  const auto beforeSystem =
      std::distance(top->getBody()->begin(), top->getBody()->end());
  expectErrorContains(test,
                      loom::hardware::rtl::deriveModuleBoundaryTransportPorts(
                          builder, builtin.roots().front().view()),
                      "Module root");
  require(test,
          std::distance(top->getBody()->begin(), top->getBody()->end()) ==
              beforeSystem,
          "wrong-root rejection produced partial CIRCT IR");

  const loom::adg::PortType tooWide =
      take(test, loom::adg::PortType::bits(mlir::IntegerType::kMaxWidth + 1U));
  loom::adg::DesignBuilder design(store);
  auto module = take(test, design.createSpatialCore("too-wide", {tooWide}, {}));
  if (llvm::Error error = module.close({}))
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  const auto beforeWidth =
      std::distance(top->getBody()->begin(), top->getBody()->end());
  expectErrorContains(test,
                      loom::hardware::rtl::deriveModuleBoundaryTransportPorts(
                          builder, finalized.roots().front().view()),
                      "integer bitwidth");
  require(test,
          std::distance(top->getBody()->begin(), top->getBody()->end()) ==
              beforeWidth,
          "width rejection produced partial CIRCT IR");
}

} // namespace

int main() {
  finalizedModuleProjectsExactTransportBoundary();
  nonModuleAndUnrepresentableWidthsFailWithoutIr();
  return 0;
}
