#include "Hardware/Implementation/RepresentationIndex.h"
#include "Common/BlobStore.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
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

ImplementationPayload putPayload(llvm::StringRef test, const BlobStore &store,
                                 PayloadRole role, llvm::StringRef name,
                                 llvm::StringRef contents) {
  return ImplementationPayload{role, name.str(),
                               take(test, store.put(bytes(contents)))};
}

std::vector<ImplementationPayload>
putSources(llvm::StringRef test, const BlobStore &store, PayloadRole role,
           std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>>
               sources) {
  std::vector<ImplementationPayload> payloads;
  payloads.reserve(sources.size());
  for (const auto &[name, contents] : sources)
    payloads.push_back(putPayload(test, store, role, name, contents));
  return take(test, canonicalizeImplementationPayloadCatalog(payloads));
}

RepresentationObjectFacts objectFacts(RepresentationObjectKind kind) {
  return RepresentationObjectFacts{kind, std::nullopt};
}

RepresentationObjectFacts portFacts(RepresentationSignalDirection direction,
                                    std::uint64_t width) {
  return RepresentationObjectFacts{
      RepresentationObjectKind::Port,
      RepresentationSignalGeometry{direction, width}};
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

void expectIndexFailure(llvm::StringRef test,
                        llvm::Expected<RepresentationIndex> value,
                        RepresentationIndexFailureKind expectedKind,
                        llvm::StringRef expectedReason) {
  if (value)
    fail(test, "accepted a representation outside the descriptor");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched = failure.kind() == expectedKind &&
                  failure.reason().contains(expectedReason);
        if (!matched)
          fail(test, failure.kind() == RepresentationIndexFailureKind::Invalid
                         ? "unexpected Invalid: " + failure.reason().str()
                         : "unexpected Unsupported: " + failure.reason().str());
      });
  if (remainder)
    fail(test, llvm::toString(std::move(remainder)));
  require(test, matched, "did not receive the expected typed index failure");
}

RepresentationFormatDescriptorRef rtlFormat(llvm::StringRef test) {
  return take(test, RepresentationFormatDescriptorRef::get(
                        RepresentationFormatKind::SystemVerilogRtl));
}

void staticDescriptorAndPureIndexApiAreClosed(
    const std::filesystem::path &root) {
  using IndexFunction = llvm::Expected<RepresentationIndex> (*)(
      RepresentationFormatDescriptorRef, const RepresentationLocator &,
      llvm::ArrayRef<ImplementationPayload>, const BlobStore &);
  static_assert(std::is_same_v<decltype(&indexRepresentation), IndexFunction>);

  const RepresentationFormatDescriptorRef rtlRef =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));

  const std::filesystem::path storePath = root / "descriptor-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const std::vector<ImplementationPayload> payloads{
      putPayload(__func__, store, PayloadRole::RtlSource, "rtl/top.sv",
                 "module top; endmodule\n")};
  const RepresentationLocator top{RepresentationObjectKind::Module, "top"};
  RepresentationIndex index =
      take(__func__, indexRepresentation(rtlRef, top, payloads, store));

  require(__func__, index.formatRef() == rtlRef,
          "index lost its descriptor owner");
  require(__func__, index.exactRoot() == top, "index lost the exact top input");
  const std::optional<RepresentationObjectFacts> topFacts =
      take(__func__, index.lookup(top));
  require(__func__, topFacts.has_value(), "exact top is absent from the index");
  require(__func__,
          *topFacts ==
              RepresentationObjectFacts{RepresentationObjectKind::Module,
                                        std::nullopt},
          "exact top has the wrong object facts");
  require(__func__, index.unresolvedExternalDefinitions().empty(),
          "closed empty top acquired an unresolved definition");
  require(__func__,
          !take(__func__, index.lookup({RepresentationObjectKind::Register,
                                        "top.absent"}))
               .has_value(),
          "exact lookup guessed an absent object");
}

void exactTopReachableRtlObjectsAreIndexed(const std::filesystem::path &root) {
  const std::filesystem::path storePath = root / "rtl-anchor-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const std::vector<ImplementationPayload> payloads =
      putSources(__func__, store, PayloadRole::RtlSource,
                 {{"rtl/00_leaf.sv",
                   R"sv(`define WIDTH 4
package unused_package;
  logic package_local;
endpackage

module leaf(input logic [`WIDTH-1:0] a,
            output logic [`WIDTH-1:0] y);
  wire [`WIDTH:0] internal_net;
  logic state;
  logic [7:0] words [0:2];
  function logic helper(input logic x);
    logic procedure_local;
    procedure_local = x;
    return procedure_local;
  endfunction
  always_comb y = a;
endmodule

module unused_root;
  missing_unused u_missing();
endmodule

module alias_port(.visible(storage));
  output storage;
endmodule

module source_name(output logic [$bits(`__FILE__)-1:0] encoded);
endmodule

module selected_port(.selected(value[0]));
  input [1:0] value;
endmodule

module selected_same_path_port(output .value(value[0]));
  wire [1:0] value;
endmodule

module converted_port(output .converted(value));
  wire [7:0] value;
endmodule
)sv"},
                  {"rtl/10_top.sv",
                   R"sv(`ifdef WIDTH
this_would_not_parse_if_macros_leaked_between_units
`endif
module top(input logic [3:0] in_data,
           output logic [3:0] out_data,
           inout wire link);
  wire internal_net;
  logic scalar_register;
  logic [11:0] packed_register;
  logic [7:0] local_memory [0:2];
  leaf u_leaf(.a(in_data), .y(out_data));
  alias_port u_alias();
  source_name u_source_name();
  selected_port u_selected();
  selected_same_path_port u_selected_same_path();
  converted_port u_converted();
  DW_fp_div #(.sig_width(4)) u_div(.a(in_data), .z(out_data));
  if (1) begin : named_scope
    logic scoped_register;
  end
endmodule

module wrapper;
  top nested();
endmodule
)sv"}});
  const RepresentationFormatDescriptorRef rtlRef =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  RepresentationIndex index = take(
      __func__,
      indexRepresentation(rtlRef, {RepresentationObjectKind::Module, "top"},
                          payloads, store));

  requireFacts(__func__, index, {RepresentationObjectKind::Module, "top"},
               objectFacts(RepresentationObjectKind::Module));
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.in_data"},
               portFacts(RepresentationSignalDirection::Input, 4));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.out_data"},
               portFacts(RepresentationSignalDirection::Output, 4));
  requireFacts(__func__, index, {RepresentationObjectKind::Port, "top.link"},
               portFacts(RepresentationSignalDirection::Inout, 1));
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Net, "top.in_data"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Register, "top.out_data"});
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.internal_net"},
               objectFacts(RepresentationObjectKind::Net));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Register, "top.scalar_register"},
               objectFacts(RepresentationObjectKind::Register));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Register, "top.packed_register"},
               objectFacts(RepresentationObjectKind::Register));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Memory, "top.local_memory"},
               objectFacts(RepresentationObjectKind::Memory));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Instance, "top.u_leaf"},
               objectFacts(RepresentationObjectKind::Instance));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_leaf.a"},
               portFacts(RepresentationSignalDirection::Input, 4));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_leaf.y"},
               portFacts(RepresentationSignalDirection::Output, 4));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.u_leaf.internal_net"},
               objectFacts(RepresentationObjectKind::Net));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Register, "top.u_leaf.state"},
               objectFacts(RepresentationObjectKind::Register));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Memory, "top.u_leaf.words"},
               objectFacts(RepresentationObjectKind::Memory));
  requireAbsent(
      __func__, index,
      {RepresentationObjectKind::Register, "top.u_leaf.procedure_local"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Register, "top.package_local"});
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_alias.visible"},
               portFacts(RepresentationSignalDirection::Output, 1));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.u_alias.storage"},
               objectFacts(RepresentationObjectKind::Net));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_source_name.encoded"},
               portFacts(RepresentationSignalDirection::Output, 112));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_selected.selected"},
               portFacts(RepresentationSignalDirection::Input, 1));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.u_selected.value"},
               objectFacts(RepresentationObjectKind::Net));
  requireFacts(
      __func__, index,
      {RepresentationObjectKind::Port, "top.u_selected_same_path.value"},
      portFacts(RepresentationSignalDirection::Output, 1));
  requireAbsent(
      __func__, index,
      {RepresentationObjectKind::Net, "top.u_selected_same_path.value"});
  requireFacts(__func__, index,
               {RepresentationObjectKind::Port, "top.u_converted.converted"},
               portFacts(RepresentationSignalDirection::Output, 8));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Net, "top.u_converted.value"},
               objectFacts(RepresentationObjectKind::Net));
  requireFacts(__func__, index,
               {RepresentationObjectKind::Instance, "top.u_div"},
               objectFacts(RepresentationObjectKind::Instance));
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Port, "top.u_div.a"});
  requireFacts(
      __func__, index,
      {RepresentationObjectKind::Register, "top.named_scope.scoped_register"},
      objectFacts(RepresentationObjectKind::Register));
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Module, "unused_root"});
  requireAbsent(__func__, index, {RepresentationObjectKind::Module, "wrapper"});
  requireAbsent(__func__, index,
                {RepresentationObjectKind::Module, "missing_unused"});
  requireFacts(__func__, index, {RepresentationObjectKind::Module, "DW_fp_div"},
               objectFacts(RepresentationObjectKind::Module));

  const std::vector<RepresentationLocator> expectedUnresolved{
      {RepresentationObjectKind::Module, "DW_fp_div"}};
  require(__func__,
          index.unresolvedExternalDefinitions() ==
              llvm::ArrayRef(expectedUnresolved),
          "unresolved definition inventory is not exact and canonical");
}

void inputClosureAndExactTopFailuresAreTyped(
    const std::filesystem::path &root) {
  const std::filesystem::path storePath = root / "input-failure-blobs";
  std::filesystem::create_directories(storePath);
  const BlobStore store(storePath.string());
  const ImplementationPayload source =
      putPayload(__func__, store, PayloadRole::RtlSource, "rtl/top.sv",
                 "module top; endmodule\n");
  const RepresentationFormatDescriptorRef format = rtlFormat(__func__);
  const RepresentationLocator top{RepresentationObjectKind::Module, "top"};

  expectIndexFailure(
      __func__,
      indexRepresentation(format, {RepresentationObjectKind::Port, "top"},
                          {source}, store),
      RepresentationIndexFailureKind::Invalid, "wrong representation");
  expectIndexFailure(
      __func__,
      indexRepresentation(format, top, llvm::ArrayRef<ImplementationPayload>(),
                          store),
      RepresentationIndexFailureKind::Invalid, "closure");

  const ImplementationPayload wrongRole =
      putPayload(__func__, store, PayloadRole::Netlist, "netlist/top.v",
                 "module top; endmodule\n");
  expectIndexFailure(__func__,
                     indexRepresentation(format, top, {wrongRole}, store),
                     RepresentationIndexFailureKind::Invalid, "role");

  const ImplementationPayload a =
      putPayload(__func__, store, PayloadRole::RtlSource, "rtl/a.sv",
                 "module a; endmodule\n");
  const ImplementationPayload z =
      putPayload(__func__, store, PayloadRole::RtlSource, "rtl/z.sv",
                 "module z; endmodule\n");
  const std::vector<ImplementationPayload> noncanonical{z, a};
  expectIndexFailure(
      __func__,
      indexRepresentation(format, {RepresentationObjectKind::Module, "a"},
                          noncanonical, store),
      RepresentationIndexFailureKind::Invalid, "canonical order");
  expectIndexFailure(
      __func__,
      indexRepresentation(format, {RepresentationObjectKind::Module, "missing"},
                          {source}, store),
      RepresentationIndexFailureKind::Invalid, "");

  const BlobDigest missingDigest = computeBlobDigest(bytes("not stored"));
  const ImplementationPayload missing{PayloadRole::RtlSource, "rtl/missing.sv",
                                      missingDigest};
  expectIndexFailure(
      __func__, indexRepresentation(format, top, {missing}, store),
      RepresentationIndexFailureKind::Invalid, "blob_store_missing");

  const std::string objectName = formatBlobDigestHex(source.blobDigest);
  std::ofstream corrupt(storePath / objectName,
                        std::ios::binary | std::ios::trunc);
  corrupt << "corrupt";
  corrupt.close();
  expectIndexFailure(
      __func__, indexRepresentation(format, top, {source}, store),
      RepresentationIndexFailureKind::Invalid, "blob_store_corruption");
}

void lexicalAndHierarchySubsetFailuresAreTyped(
    const std::filesystem::path &root) {
  const RepresentationFormatDescriptorRef format = rtlFormat(__func__);
  const RepresentationLocator top{RepresentationObjectKind::Module, "top"};
  auto expectUnsupported = [&](llvm::StringRef name, llvm::StringRef source) {
    const std::filesystem::path storePath = root / name.str();
    std::filesystem::create_directories(storePath);
    const BlobStore store(storePath.string());
    const std::vector<ImplementationPayload> payloads = putSources(
        __func__, store, PayloadRole::RtlSource, {{"rtl/top.sv", source}});
    expectIndexFailure(name, indexRepresentation(format, top, payloads, store),
                       RepresentationIndexFailureKind::Unsupported, "");
  };

  expectUnsupported("active-include-blobs",
                    "`include \"missing.svh\"\nmodule top; endmodule\n");
  expectUnsupported("inactive-include-blobs",
                    "`ifdef NEVER_DEFINED\n`include \"missing.svh\"\n`endif\n"
                    "module top; endmodule\n");
  expectUnsupported("escaped-name-blobs",
                    "module top; logic \\escaped.name ; endmodule\n");
  expectUnsupported("keyword-mode-directive-blobs",
                    "`begin_keywords \"1800-2005\"\nmodule top; endmodule\n"
                    "`end_keywords\n");
  expectUnsupported("pragma-directive-blobs",
                    "`pragma protect begin_protected\nmodule top; endmodule\n");
  expectUnsupported("min-typ-max-blobs",
                    "module top(input logic [(1:3:7)-1:0] value); endmodule\n");
  expectUnsupported("implicit-generate-blobs",
                    "module top; if (1) begin logic state; end endmodule\n");
  expectUnsupported("generate-array-blobs",
                    "module top; for (genvar i = 0; i < 2; i++) begin : g "
                    "logic state; end endmodule\n");
  expectUnsupported(
      "resolved-instance-array-blobs",
      "module leaf; endmodule\nmodule top; leaf u[1:0](); endmodule\n");
  expectUnsupported("unknown-instance-array-blobs",
                    "module top; missing u[1:0](); endmodule\n");
  expectUnsupported("dynamic-variable-blobs",
                    "module top; logic values[]; endmodule\n");
  expectUnsupported("queue-variable-blobs",
                    "module top; logic values[$]; endmodule\n");
  expectUnsupported("associative-variable-blobs",
                    "module top; logic values[int]; endmodule\n");
  expectUnsupported(
      "combined-port-blobs",
      "module top(.pair({left, right})); input left, right; endmodule\n");
  expectUnsupported("reference-port-blobs",
                    "module top(ref logic value); endmodule\n");
  expectUnsupported("unpacked-port-blobs",
                    "module top(input logic value[1:0]); endmodule\n");
  expectUnsupported("class-declaration-blobs",
                    "module top; class local_class; endclass endmodule\n");
  expectUnsupported("interface-top-blobs", "interface top; endinterface\n");
  expectUnsupported("program-top-blobs", "program top; endprogram\n");
  expectUnsupported("unnamed-instance-blobs",
                    "module leaf(input logic a); endmodule\n"
                    "module top(input logic a); leaf(a); endmodule\n");

  expectUnsupported(
      "unused-interface-blobs",
      "module top; endmodule\ninterface unused_interface; endinterface\n");
  expectUnsupported(
      "unused-program-blobs",
      "module top; endmodule\nprogram unused_program; endprogram\n");
  expectUnsupported(
      "unused-checker-blobs",
      "module top; endmodule\nchecker unused_checker; endchecker\n");
  expectUnsupported("unused-class-blobs",
                    "module top; endmodule\nclass unused_class; endclass\n");
  expectUnsupported(
      "unused-instance-array-blobs",
      "module leaf; endmodule\n"
      "module unused; leaf u[1:0](); endmodule\nmodule top; endmodule\n");
  expectUnsupported("unused-implicit-generate-blobs",
                    "module unused; if (1) begin logic state; end endmodule\n"
                    "module top; endmodule\n");
}

void frontendFailuresRemainRecoverableAndWarningsNonsemantic(
    const std::filesystem::path &root) {
  const RepresentationFormatDescriptorRef format = rtlFormat(__func__);
  const RepresentationLocator top{RepresentationObjectKind::Module, "top"};
  auto expectInvalid = [&](llvm::StringRef name, llvm::StringRef source) {
    const std::filesystem::path storePath = root / name.str();
    std::filesystem::create_directories(storePath);
    const BlobStore store(storePath.string());
    const std::vector<ImplementationPayload> payloads = putSources(
        __func__, store, PayloadRole::RtlSource, {{"rtl/top.sv", source}});
    expectIndexFailure(name, indexRepresentation(format, top, payloads, store),
                       RepresentationIndexFailureKind::Invalid,
                       "parse or elaboration");
  };

  expectInvalid("malformed-source-blobs",
                "module top; this is not SystemVerilog endmodule\n");
  expectInvalid(
      "unused-semantic-error-blobs",
      "module top; endmodule\n"
      "module unused; logic value; assign value = missing_name; endmodule\n");

  auto buildVariant = [&](llvm::StringRef storeName, llvm::StringRef source) {
    const std::filesystem::path storePath = root / storeName.str();
    std::filesystem::create_directories(storePath);
    const BlobStore store(storePath.string());
    const std::vector<ImplementationPayload> payloads = putSources(
        __func__, store, PayloadRole::RtlSource, {{"rtl/design.sv", source}});
    return take(__func__, indexRepresentation(format, top, payloads, store));
  };
  RepresentationIndex first = buildVariant(
      "determinism-a-blobs",
      "module child; wire nested; endmodule\n"
      "module top; logic [9:0] value = 9000; missing_b b(); child u(); "
      "missing_a a(); missing_a a2(); endmodule\n");
  RepresentationIndex second =
      buildVariant("determinism-b-blobs",
                   "module top; missing_a a(); child u(); missing_b b(); "
                   "logic [9:0] value = 10'd808; missing_a a2(); endmodule\n"
                   "module child; wire nested; endmodule\n");
  for (const RepresentationLocator &locator :
       std::vector<RepresentationLocator>{
           {RepresentationObjectKind::Module, "top"},
           {RepresentationObjectKind::Register, "top.value"},
           {RepresentationObjectKind::Instance, "top.a"},
           {RepresentationObjectKind::Instance, "top.b"},
           {RepresentationObjectKind::Instance, "top.u"},
           {RepresentationObjectKind::Net, "top.u.nested"}}) {
    const std::optional<RepresentationObjectFacts> lhs =
        take(__func__, first.lookup(locator));
    const std::optional<RepresentationObjectFacts> rhs =
        take(__func__, second.lookup(locator));
    require(__func__, lhs == rhs,
            "authoring order or warning changed an indexed fact");
  }
  require(__func__,
          first.unresolvedExternalDefinitions() ==
              second.unresolvedExternalDefinitions(),
          "authoring order changed unresolved-definition canonicalization");
  const std::vector<RepresentationLocator> expectedUnresolved{
      {RepresentationObjectKind::Module, "missing_a"},
      {RepresentationObjectKind::Module, "missing_b"}};
  require(__func__,
          first.unresolvedExternalDefinitions() ==
              llvm::ArrayRef(expectedUnresolved),
          "unresolved definitions are not sorted and unique");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one ignored test-root argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  staticDescriptorAndPureIndexApiAreClosed(root);
  exactTopReachableRtlObjectsAreIndexed(root);
  inputClosureAndExactTopFailuresAreTyped(root);
  lexicalAndHierarchySubsetFailuresAreTyped(root);
  frontendFailuresRemainRecoverableAndWarningsNonsemantic(root);
  std::filesystem::remove_all(root);
  return EXIT_SUCCESS;
}
