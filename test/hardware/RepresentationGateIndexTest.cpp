#include "Common/BlobStore.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::hardware;

namespace {

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

std::vector<std::uint8_t> bytes(llvm::StringRef text) {
  return {text.bytes_begin(), text.bytes_end()};
}

std::vector<ImplementationPayload>
putSources(llvm::StringRef test, const BlobStore &store,
           std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>>
               sources) {
  std::vector<ImplementationPayload> payloads;
  payloads.reserve(sources.size());
  for (const auto &[name, contents] : sources) {
    payloads.push_back(
        ImplementationPayload{PayloadRole::Netlist, name.str(),
                              take(test, store.put(bytes(contents)))});
  }
  return take(test, canonicalizeImplementationPayloadCatalog(payloads));
}

RepresentationFormatDescriptorRef gateFormat(llvm::StringRef test) {
  return take(test,
              RepresentationFormatDescriptorRef::get(
                  RepresentationFormatKind::StructuralVerilogGateNetlist));
}

RepresentationObjectFacts objectFacts(RepresentationObjectKind kind) {
  return RepresentationObjectFacts{kind, std::nullopt};
}

RepresentationObjectFacts terminalFacts(RepresentationObjectKind kind,
                                        RepresentationSignalDirection direction,
                                        std::uint64_t width) {
  return RepresentationObjectFacts{
      kind, RepresentationSignalGeometry{direction, width}};
}

void requireFacts(llvm::StringRef test, const RepresentationIndex &index,
                  RepresentationLocator locator,
                  RepresentationObjectFacts expected) {
  const std::optional<RepresentationObjectFacts> actual =
      take(test, index.lookup(locator));
  require(test, actual.has_value(),
          "missing representation object '" + locator.canonicalName + "'");
  require(test, *actual == expected,
          "wrong facts for representation object '" + locator.canonicalName +
              "'");
}

void requireAbsent(llvm::StringRef test, const RepresentationIndex &index,
                   RepresentationLocator locator) {
  require(test, !take(test, index.lookup(locator)).has_value(),
          "unexpected representation object '" + locator.canonicalName + "'");
}

void expectUnsupported(llvm::StringRef test,
                       llvm::Expected<RepresentationIndex> value) {
  if (value)
    fail(test, "accepted syntax outside the structural gate descriptor");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched = failure.kind() == RepresentationIndexFailureKind::Unsupported;
        if (!matched)
          fail(test,
               "expected Unsupported but received: " + failure.reason().str());
      });
  if (remainder)
    fail(test, llvm::toString(std::move(remainder)));
  require(test, matched, "did not receive a typed Unsupported failure");
}

void expectInvalid(llvm::StringRef test,
                   llvm::Expected<RepresentationIndex> value) {
  if (value)
    fail(test, "accepted source outside the selected language profile");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched = failure.kind() == RepresentationIndexFailureKind::Invalid;
        if (!matched)
          fail(test,
               "expected Invalid but received: " + failure.reason().str());
      });
  if (remainder)
    fail(test, llvm::toString(std::move(remainder)));
  require(test, matched, "did not receive a typed Invalid failure");
}

llvm::Expected<RepresentationIndex>
tryBuildGateIndex(const std::filesystem::path &root, llvm::StringRef storeName,
                  llvm::StringRef source) {
  const std::filesystem::path storePath = root / storeName.str();
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const std::vector<ImplementationPayload> payloads =
      putSources(__func__, store, {{"netlist/design.v", source}});
  return indexRepresentation(gateFormat(__func__),
                             {RepresentationObjectKind::Module, "top"},
                             payloads, store);
}

llvm::Expected<RepresentationIndex>
tryBuildGateIndexFromUnits(
    const std::filesystem::path &root, llvm::StringRef storeName,
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> units) {
  const std::filesystem::path storePath = root / storeName.str();
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const std::vector<ImplementationPayload> payloads =
      putSources(__func__, store, units);
  return indexRepresentation(gateFormat(__func__),
                             {RepresentationObjectKind::Module, "top"},
                             payloads, store);
}

RepresentationIndex buildGateIndex(const std::filesystem::path &root,
                                   llvm::StringRef storeName,
                                   llvm::StringRef source) {
  return take(storeName, tryBuildGateIndex(root, storeName, source));
}

void completeStructuralGateSubsetIsIndexed(const std::filesystem::path &root) {
  RepresentationIndex index = buildGateIndex(root, "gate-anchor-blobs",
                                             R"v(primitive udp_buffer(out, in);
  output out;
  input in;
  table
    0 : 0;
    1 : 1;
    x : x;
  endtable
endprimitive

module leaf(a, y, io);
  input a;
  output y;
  inout io;
  wire internal;
  assign internal = a;
  assign y = internal;
endmodule

module top(a, y, io);
  input [3:0] a;
  output [3:0] y;
  inout io;
  wire [3:0] bus;
  wire selected;
  wire [1:0] slice;
  wire [1:0] repeated;
  wire initialized = a[0];
  leaf u_leaf(.a(a[0]), .y(y[0]), .io(io));
  udp_buffer u_udp(y[1], a[1]);
  NAND2_X1 u1(.a(a[2]), .z(y[2]));
  generate
    if (1) begin : named_scope
      leaf u_nested(.a(a[3]), .y(y[3]), .io(io));
    end
  endgenerate
  assign bus = {a[1:0], 2'b10};
  assign selected = bus[0];
  assign slice = bus[3:2];
  assign repeated = {2{a[0]}};
endmodule
)v");

  requireFacts(__func__, index, {RepresentationObjectKind::Module, "top"},
               objectFacts(RepresentationObjectKind::Module));
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.a"},
               terminalFacts(RepresentationObjectKind::Port,
                             RepresentationSignalDirection::Input, 4));
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.y"},
               terminalFacts(RepresentationObjectKind::Port,
                             RepresentationSignalDirection::Output, 4));
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.io"},
               terminalFacts(RepresentationObjectKind::Port,
                             RepresentationSignalDirection::Inout, 1));
  requireAbsent(__func__, index, {RepresentationObjectKind::Net, "top.a"});
  requireAbsent(__func__, index, {RepresentationObjectKind::Net, "top.y"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Net, "top.u_leaf.a"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Net, "top.u_leaf.y"});

  for (llvm::StringRef name :
       {"bus", "selected", "slice", "repeated", "initialized"})
    requireFacts(__func__, index,
                 {RepresentationObjectKind::Net, ("top." + name).str()},
                 objectFacts(RepresentationObjectKind::Net));

  requireFacts(__func__, index, {RepresentationObjectKind::Cell, "top.u_leaf"},
               objectFacts(RepresentationObjectKind::Cell));
  requireFacts(__func__, index, {RepresentationObjectKind::Pin, "top.u_leaf.a"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Input, 1));
  requireFacts(__func__, index, {RepresentationObjectKind::Pin, "top.u_leaf.y"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Output, 1));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Pin, "top.u_leaf.io"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Inout, 1));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.u_leaf.internal"},
               objectFacts(RepresentationObjectKind::Net));

  requireFacts(__func__, index, {RepresentationObjectKind::Cell, "top.u_udp"},
               objectFacts(RepresentationObjectKind::Cell));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Pin, "top.u_udp.out"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Output, 1));
  requireFacts(__func__, index, {RepresentationObjectKind::Pin, "top.u_udp.in"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Input, 1));

  requireFacts(__func__, index, {RepresentationObjectKind::Cell, "top.u1"},
               objectFacts(RepresentationObjectKind::Cell));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Module, "NAND2_X1"},
               objectFacts(RepresentationObjectKind::Module));
  requireAbsent(__func__, index, {RepresentationObjectKind::Pin, "top.u1.a"});
  requireAbsent(__func__, index, {RepresentationObjectKind::Pin, "top.u1.z"});

  requireFacts(__func__, index,
               {RepresentationObjectKind::Cell, "top.named_scope.u_nested"},
               objectFacts(RepresentationObjectKind::Cell));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Pin, "top.named_scope.u_nested.a"},
               terminalFacts(RepresentationObjectKind::Pin,
                             RepresentationSignalDirection::Input, 1));

  const std::vector<RepresentationLocator> expectedUnresolved{
      {RepresentationObjectKind::Module, "NAND2_X1"}};
  require(__func__,
          index.unresolvedExternalDefinitions() ==
              llvm::ArrayRef(expectedUnresolved),
          "unknown cell inventory is not exact and canonical");
}

void wiringGrammarIsUniformAcrossElectricalSites(
    const std::filesystem::path &root) {
  RepresentationIndex index =
      buildGateIndex(root, "gate-wiring-expression-blobs",
                     R"v(primitive udp_buffer(out, in);
  output out;
  input in;
  table
    0 : 0;
    1 : 1;
    x : x;
  endtable
endprimitive

module leaf(a, y);
  input [3:0] a;
  output [1:0] y;
endmodule

module top(a, y);
  input [3:0] a;
  output [3:0] y;
  wire [1:0] initialized = ({a[1], a[0]});
  wire [1:0] connected;
  leaf u_leaf(.a({a[3:2], 2'b10}), .y(connected[1:0]));
  udp_buffer u_udp(y[0], (a[0]));
  missing_cell u_missing(.assert({2{a[1]}}), .property(1'b0));
  assign {y[2], y[1]} = ({initialized[1], initialized[0]});
endmodule
)v");

  for (llvm::StringRef name : {"u_leaf", "u_udp", "u_missing"})
    requireFacts(__func__, index,
                 {RepresentationObjectKind::Cell, ("top." + name).str()},
                 objectFacts(RepresentationObjectKind::Cell));
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Pin, "top.u_missing.assert"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Pin, "top.u_missing.property"});
}

void behavioralCellActualsAreUnsupported(const std::filesystem::path &root) {
  auto reject = [&](llvm::StringRef name, llvm::StringRef source) {
    expectUnsupported(name, tryBuildGateIndex(root, name, source));
  };

  reject("gate-resolved-cell-actual-blobs",
         "module leaf(input a); endmodule\n"
         "module top(input a, b); leaf u(.a(a + b)); endmodule\n");
  reject("gate-udp-cell-actual-blobs",
         "primitive udp_buffer(out, in); output out; input in; "
         "table 0 : 0; 1 : 1; endtable endprimitive\n"
         "module top(input a, b, output y); "
         "udp_buffer u(y, a & b); endmodule\n");
  reject("gate-unknown-cell-actual-blobs",
         "module top(input a, b); missing_cell u(.a(a == b)); endmodule\n");
}

void assignmentsAreOnlyUnpackedAtContinuousAssignSites(
    const std::filesystem::path &root) {
  auto reject = [&](llvm::StringRef name, llvm::StringRef source) {
    expectUnsupported(name, tryBuildGateIndex(root, name, source));
  };

  reject("gate-assignment-in-cell-actual-blobs",
         "module leaf(input a); endmodule\n"
         "module top(input a, output y); leaf u(.a(y = a)); endmodule\n");
  reject("gate-assignment-in-net-initializer-blobs",
         "module top(input a, output y); wire w = (y = a); endmodule\n");
  reject("gate-operator-nested-in-concatenation-blobs",
         "module top(input a, b, output [1:0] y); "
         "assign y = {a & b, 1'b0}; endmodule\n");
}

void validityIsEstablishedOverTheWholeClosure(
    const std::filesystem::path &root) {
  expectInvalid(
      "gate-mixed-failure-one-unit-blobs",
      tryBuildGateIndex(root, "gate-mixed-failure-one-unit-blobs",
                        "module top(input a, b, output y, output z); "
                        "assign y = a & b; "
                        "assign z = ; endmodule\n"));

  expectInvalid("gate-mixed-failure-across-units-blobs",
                tryBuildGateIndexFromUnits(
                    root, "gate-mixed-failure-across-units-blobs",
                    {{"netlist/a_subset.v",
                      "module top(input a, b, output y); "
                      "assign y = a & b; endmodule\n"},
                     {"netlist/b_invalid.v",
                      "module other(input a); assign = a; endmodule\n"}}));

  expectInvalid("gate-raw-directive-does-not-mask-parse-error-blobs",
                tryBuildGateIndexFromUnits(
                    root, "gate-raw-directive-does-not-mask-parse-error-blobs",
                    {{"netlist/a_directive.v",
                      "`timescale 1ns/1ps\n"
                      "module helper(input a); endmodule\n"},
                     {"netlist/b_invalid.v",
                      "module other(input a); assign = a; endmodule\n"}}));

  expectInvalid("gate-exact-top-does-not-mask-parse-error-blobs",
                tryBuildGateIndexFromUnits(
                    root, "gate-exact-top-does-not-mask-parse-error-blobs",
                    {{"netlist/a_no_top.v",
                      "module helper(input a); endmodule\n"},
                     {"netlist/b_invalid.v",
                      "module other(input a); assign = a; endmodule\n"}}));
}

void gateLanguageValidityPrecedesSubsetAdmission(
    const std::filesystem::path &root) {
  expectInvalid(
      "gate-systemverilog-assertion-blobs",
      tryBuildGateIndex(root, "gate-systemverilog-assertion-blobs",
                        "module top(input clk, a); "
                        "assert property (@(posedge clk) a); endmodule\n"));

  RepresentationIndex index = buildGateIndex(
      root, "gate-assert-property-identifiers-blobs",
      "module top(assert, property); input assert; output property; "
      "assign property = assert; endmodule\n");
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.assert"},
               terminalFacts(RepresentationObjectKind::Port,
                             RepresentationSignalDirection::Input, 1));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.property"},
               terminalFacts(RepresentationObjectKind::Port,
                             RepresentationSignalDirection::Output, 1));
}

void structuralGateRejectionsCoverTheWholePayload(
    const std::filesystem::path &root) {
  auto reject = [&](llvm::StringRef name, llvm::StringRef source) {
    const std::filesystem::path storePath = root / name.str();
    std::filesystem::create_directories(storePath);
    const BlobStore store(storePath.string());
    const std::vector<ImplementationPayload> payloads =
        putSources(__func__, store, {{"netlist/design.v", source}});
    expectUnsupported(
        name, indexRepresentation(gateFormat(__func__),
                                  {RepresentationObjectKind::Module, "top"},
                                  payloads, store));
  };

  reject("gate-procedure-blobs",
         "module top(input a, output reg y); always @* y = a; endmodule\n");
  reject("gate-timing-blobs",
         "module top(input a, output y); assign #1 y = a; endmodule\n");
  reject("gate-runtime-variable-blobs", "module top; reg state; endmodule\n");
  reject("gate-memory-blobs", "module top; reg values [0:1]; endmodule\n");
  reject("gate-subroutine-blobs",
         "module top; function value; input value; value = 1'b0; "
         "endfunction endmodule\n");
  reject("gate-arithmetic-blobs",
         "module top(input a, b, output y); assign y = a + b; endmodule\n");
  reject("gate-bitwise-blobs",
         "module top(input a, b, output y); assign y = a & b; endmodule\n");
  reject("gate-comparison-blobs",
         "module top(input a, b, output y); assign y = a == b; endmodule\n");
  reject("gate-logical-blobs",
         "module top(input a, b, output y); assign y = a && b; endmodule\n");
  reject("gate-conditional-blobs",
         "module top(input a, b, c, output y); assign y = a ? b : c; "
         "endmodule\n");
  reject("gate-net-initializer-operator-blobs",
         "module top(input a, b); wire y = a & b; endmodule\n");
  reject("gate-builtin-primitive-blobs",
         "module top(input a, b, output y); and g(y, a, b); endmodule\n");
  reject("gate-switch-primitive-blobs",
         "module top(inout a, b); tran t(a, b); endmodule\n");
  reject("gate-unnamed-instance-blobs",
         "module leaf; endmodule\nmodule top; leaf (); endmodule\n");
  reject("gate-arrayed-instance-blobs",
         "module leaf; endmodule\nmodule top; leaf u[1:0](); endmodule\n");
  reject("gate-implicit-generate-blobs",
         "module top; generate if (1) begin wire value; end endgenerate "
         "endmodule\n");
  reject("gate-unused-arithmetic-blobs",
         "module unused(input a, b, output y); assign y = a + b; endmodule\n"
         "module top; endmodule\n");
  reject("gate-unused-arrayed-instance-blobs",
         "module leaf; endmodule\n"
         "module unused; leaf u[1:0](); endmodule\nmodule top; endmodule\n");
  reject("gate-unused-implicit-generate-blobs",
         "module unused; generate if (1) begin wire value; end endgenerate "
         "endmodule\nmodule top; endmodule\n");
}

void warningsAndAuthoringOrderAreNonsemantic(
    const std::filesystem::path &root) {
  RepresentationIndex first = buildGateIndex(
      root, "gate-order-a-blobs",
      "module leaf(input a, output y); wire n; assign n = a; assign y = n; "
      "endmodule\n"
      "module top; wire [1:0] value; missing_b b(); leaf u(); "
      "missing_a a(); assign value = 4'b1111; endmodule\n");
  RepresentationIndex second = buildGateIndex(
      root, "gate-order-b-blobs",
      "module top; missing_a a(); wire [1:0] value; leaf u(); missing_b b(); "
      "assign value = 2'b11; endmodule\n"
      "module leaf(input a, output y); assign y = a; wire n; endmodule\n");

  for (const RepresentationLocator &locator :
       std::vector<RepresentationLocator>{
           {RepresentationObjectKind::Module, "top"},
           {RepresentationObjectKind::Net, "top.value"},
           {RepresentationObjectKind::Cell, "top.a"},
           {RepresentationObjectKind::Cell, "top.b"},
           {RepresentationObjectKind::Cell, "top.u"},
           {RepresentationObjectKind::Pin, "top.u.a"},
           {RepresentationObjectKind::Pin, "top.u.y"},
           {RepresentationObjectKind::Net, "top.u.n"}}) {
    require(__func__,
            take(__func__, first.lookup(locator)) ==
                take(__func__, second.lookup(locator)),
            "warning or authoring order changed indexed gate facts");
  }
  require(__func__,
          first.unresolvedExternalDefinitions() ==
              second.unresolvedExternalDefinitions(),
          "authoring order changed unresolved gate definitions");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one ignored test-root argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  completeStructuralGateSubsetIsIndexed(root);
  wiringGrammarIsUniformAcrossElectricalSites(root);
  behavioralCellActualsAreUnsupported(root);
  assignmentsAreOnlyUnpackedAtContinuousAssignSites(root);
  gateLanguageValidityPrecedesSubsetAdmission(root);
  validityIsEstablishedOverTheWholeClosure(root);
  structuralGateRejectionsCoverTheWholePayload(root);
  warningsAndAuthoringOrderAreNonsemantic(root);
  std::filesystem::remove_all(root);
  return EXIT_SUCCESS;
}
