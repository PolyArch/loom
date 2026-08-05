#include "Hardware/Implementation/RepresentationFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid representation format reference");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t>
expectedBinaryReference(RepresentationFormatKind kind) {
  constexpr llvm::StringLiteral identity =
      "loom.hardware_representation_format";
  std::vector<std::uint8_t> expected{0, 0, 0, 0, 0, 0, 0, 35};
  expected.insert(expected.end(), identity.bytes_begin(), identity.bytes_end());
  const std::vector<std::uint8_t> suffix{
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, static_cast<std::uint8_t>(kind),
  };
  expected.insert(expected.end(), suffix.begin(), suffix.end());
  return expected;
}

void exactBinaryCodecIsClosed() {
  const auto rtl =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const auto netlist = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));

  require(__func__,
          hardwareRepresentationFormatRegistry.identity ==
              "loom.hardware_representation_format",
          "registry identity changed");
  require(__func__,
          hardwareRepresentationFormatRegistry.version ==
              loom::SchemaVersion{1, 0},
          "registry version changed");
  require(__func__, rtl.kind() == RepresentationFormatKind::SystemVerilogRtl,
          "RTL kind changed");
  require(__func__,
          netlist.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist,
          "gate-netlist kind changed");

  const std::vector<std::uint8_t> rtlBytes =
      encodeRepresentationFormatDescriptorRef(rtl);
  const std::vector<std::uint8_t> netlistBytes =
      encodeRepresentationFormatDescriptorRef(netlist);
  require(__func__, rtlBytes == expectedBinaryReference(rtl.kind()),
          "RTL reference bytes changed");
  require(__func__, netlistBytes == expectedBinaryReference(netlist.kind()),
          "gate-netlist reference bytes changed");
  require(__func__, rtlBytes.size() == 55,
          "reference framing has the wrong size");
  require(__func__,
          take(__func__, decodeRepresentationFormatDescriptorRef(rtlBytes)) ==
              rtl,
          "RTL binary reference did not round-trip");
  require(__func__,
          take(__func__, decodeRepresentationFormatDescriptorRef(
                             netlistBytes)) == netlist,
          "gate-netlist binary reference did not round-trip");

  expectError(__func__,
              RepresentationFormatDescriptorRef::get(
                  static_cast<RepresentationFormatKind>(2)),
              "kind");

  for (std::size_t size = 0; size < rtlBytes.size(); ++size)
    expectError(__func__,
                decodeRepresentationFormatDescriptorRef(
                    llvm::ArrayRef(rtlBytes).take_front(size)),
                "truncated");

  std::vector<std::uint8_t> trailing = rtlBytes;
  trailing.push_back(0);
  expectError(__func__, decodeRepresentationFormatDescriptorRef(trailing),
              "trailing");

  std::vector<std::uint8_t> wrongRegistry = rtlBytes;
  wrongRegistry[8] = 'L';
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongRegistry),
              "registry");

  std::vector<std::uint8_t> wrongVersion = rtlBytes;
  wrongVersion[46] = 2;
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongVersion),
              "version");

  std::vector<std::uint8_t> wrongKind = rtlBytes;
  wrongKind[54] = 2;
  expectError(__func__, decodeRepresentationFormatDescriptorRef(wrongKind),
              "kind");
}

void exactJsonCodecIsClosed() {
  const auto rtl =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const auto netlist = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));
  constexpr llvm::StringLiteral rtlJson =
      R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":0})json";
  constexpr llvm::StringLiteral netlistJson =
      R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":1})json";

  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(rtl) == rtlJson,
          "RTL canonical JSON changed");
  require(__func__,
          serializeRepresentationFormatDescriptorRefJson(netlist) ==
              netlistJson,
          "gate-netlist canonical JSON changed");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(rtlJson)) ==
              rtl,
          "RTL JSON reference did not round-trip");
  require(__func__,
          take(__func__, parseRepresentationFormatDescriptorRefJson(
                             netlistJson)) == netlist,
          "gate-netlist JSON reference did not round-trip");

  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"major":1,"registry":"loom.hardware_representation_format","minor":0,"kind":0})json"),
      "canonical");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":0,"name":"sv"})json"),
      "field");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"other","major":1,"minor":0,"kind":0})json"),
      "registry");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":1,"minor":1,"kind":0})json"),
      "version");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":2})json"),
      "kind");
  expectError(
      __func__,
      parseRepresentationFormatDescriptorRefJson(
          R"json({"registry":"loom.hardware_representation_format","major":1,"minor":0,"kind":-1})json"),
      "unsigned");
}

} // namespace

int main() {
  exactBinaryCodecIsClosed();
  exactJsonCodecIsClosed();
  return EXIT_SUCCESS;
}
