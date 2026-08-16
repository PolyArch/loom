#include "Hardware/Implementation/RepresentationLocator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
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
    fail(test, "accepted an invalid representation locator");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted an invalid representation locator");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t> expectedBytes(std::uint32_t kind,
                                        llvm::StringRef name) {
  std::vector<std::uint8_t> bytes{
      static_cast<std::uint8_t>(kind >> 24),
      static_cast<std::uint8_t>(kind >> 16),
      static_cast<std::uint8_t>(kind >> 8),
      static_cast<std::uint8_t>(kind),
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      static_cast<std::uint8_t>(name.size()),
  };
  bytes.insert(bytes.end(), name.bytes_begin(), name.bytes_end());
  return bytes;
}

void exactBinaryCodecIsClosed() {
  constexpr std::array<RepresentationObjectKind, 11> kinds{
      RepresentationObjectKind::Module,
      RepresentationObjectKind::Instance,
      RepresentationObjectKind::Port,
      RepresentationObjectKind::Net,
      RepresentationObjectKind::Register,
      RepresentationObjectKind::Memory,
      RepresentationObjectKind::Cell,
      RepresentationObjectKind::Pin,
      RepresentationObjectKind::PhysicalObject,
      RepresentationObjectKind::DeviceResource,
      RepresentationObjectKind::Model,
  };
  for (std::uint32_t tag = 0; tag < kinds.size(); ++tag) {
    const RepresentationLocator locator{kinds[tag], "top.object"};
    const std::vector<std::uint8_t> bytes =
        take(__func__, encodeRepresentationLocator(locator));
    require(__func__, bytes == expectedBytes(tag, locator.canonicalName),
            "locator binary framing changed");
    require(__func__,
            take(__func__, decodeRepresentationLocator(bytes)) == locator,
            "locator binary codec did not round-trip");
  }

  const RepresentationLocator module{RepresentationObjectKind::Module, "top"};
  const std::vector<std::uint8_t> bytes =
      take(__func__, encodeRepresentationLocator(module));
  for (std::size_t size = 0; size < bytes.size(); ++size)
    expectError(
        __func__,
        decodeRepresentationLocator(llvm::ArrayRef(bytes).take_front(size)),
        "truncated");

  std::vector<std::uint8_t> trailing = bytes;
  trailing.push_back(0);
  expectError(__func__, decodeRepresentationLocator(trailing), "trailing");

  std::vector<std::uint8_t> unknownKind = bytes;
  unknownKind[3] = 11;
  expectError(__func__, decodeRepresentationLocator(unknownKind), "kind");
  expectError(__func__,
              encodeRepresentationLocator(
                  {static_cast<RepresentationObjectKind>(11), "top"}),
              "kind");

  std::vector<std::uint8_t> excessiveLength = bytes;
  std::fill(excessiveLength.begin() + 4, excessiveLength.begin() + 12, 0xff);
  expectError(__func__, decodeRepresentationLocator(excessiveLength),
              "truncated");

  const RepresentationLocator longName{RepresentationObjectKind::Net,
                                       std::string(256, 'a')};
  const std::vector<std::uint8_t> longBytes =
      take(__func__, encodeRepresentationLocator(longName));
  require(__func__,
          longBytes.size() == 268 && longBytes[10] == 1 && longBytes[11] == 0,
          "locator name length is not full-width u64be");
  require(__func__,
          take(__func__, decodeRepresentationLocator(longBytes)) == longName,
          "multi-byte locator name length did not round-trip");
}

void exactJsonCodecIsClosed() {
  constexpr std::array<llvm::StringLiteral, 11> spellings{
      "Module",         "Instance",       "Port", "Net",
      "Register",       "Memory",         "Cell", "Pin",
      "PhysicalObject", "DeviceResource", "Model",
  };
  for (std::uint32_t tag = 0; tag < spellings.size(); ++tag) {
    const RepresentationLocator locator{
        static_cast<RepresentationObjectKind>(tag), "top.object"};
    const std::string expected =
        (llvm::Twine("{\"object_kind\":\"") + spellings[tag] +
         "\",\"canonical_name\":\"top.object\"}")
            .str();
    const std::string json =
        take(__func__, serializeRepresentationLocatorJson(locator));
    require(__func__, json == expected, "locator canonical JSON changed");
    require(__func__,
            take(__func__, parseRepresentationLocatorJson(json)) == locator,
            "locator JSON codec did not round-trip");
  }

  expectError(__func__,
              parseRepresentationLocatorJson(
                  R"json({"canonical_name":"top","object_kind":"Module"})json"),
              "canonical");
  expectError(
      __func__,
      parseRepresentationLocatorJson(
          R"json({"object_kind":"Module","canonical_name":"top","extra":0})json"),
      "field");
  expectError(__func__,
              parseRepresentationLocatorJson(
                  R"json({"object_kind":"module","canonical_name":"top"})json"),
              "kind");
  expectError(
      __func__,
      parseRepresentationLocatorJson(
          R"json({"object_kind":"\u004dodule","canonical_name":"top"})json"),
      "canonical");
  expectError(__func__,
              serializeRepresentationLocatorJson(
                  {static_cast<RepresentationObjectKind>(11), "top"}),
              "kind");
}

void canonicalOrderingUsesEncodedBytes() {
  std::vector<RepresentationLocator> locators{
      {RepresentationObjectKind::Instance, "a"},
      {RepresentationObjectKind::Module, "aa"},
      {RepresentationObjectKind::Module, "z"},
      {RepresentationObjectKind::Module, "ab"},
  };
  std::sort(locators.begin(), locators.end(),
            representationLocatorCanonicalLess);
  const std::vector<RepresentationLocator> expected{
      {RepresentationObjectKind::Module, "z"},
      {RepresentationObjectKind::Module, "aa"},
      {RepresentationObjectKind::Module, "ab"},
      {RepresentationObjectKind::Instance, "a"},
  };
  require(__func__, locators == expected,
          "locator ordering does not follow canonical bytes");

  const RepresentationLocator highByteA{
      RepresentationObjectKind::Net, std::string(1, static_cast<char>(0x80))};
  const RepresentationLocator highByteB{
      RepresentationObjectKind::Net, std::string(1, static_cast<char>(0xff))};
  require(__func__,
          representationLocatorCanonicalLess(highByteA, highByteB) &&
              !representationLocatorCanonicalLess(highByteB, highByteA),
          "locator ordering does not compare name bytes as unsigned bytes");
}

void initialFormatsOwnHdlSyntax() {
  const auto rtl =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  const auto gate = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));

  for (const RepresentationLocator &locator :
       std::array<RepresentationLocator, 8>{
           RepresentationLocator{RepresentationObjectKind::Module, "top"},
           RepresentationLocator{RepresentationObjectKind::Instance, "top.u$0"},
           RepresentationLocator{RepresentationObjectKind::Port, "top.a"},
           RepresentationLocator{RepresentationObjectKind::Port,
                                 "top.u$0.result"},
           RepresentationLocator{RepresentationObjectKind::Net, "top._n"},
           RepresentationLocator{RepresentationObjectKind::Register,
                                 "top.state"},
           RepresentationLocator{RepresentationObjectKind::Memory,
                                 "top.buffer"},
           RepresentationLocator{RepresentationObjectKind::Module, "_top"},
       })
    if (llvm::Error error = validateRepresentationLocatorSyntax(rtl, locator))
      fail(__func__, llvm::toString(std::move(error)));

  for (const RepresentationLocator &locator :
       std::array<RepresentationLocator, 5>{
           RepresentationLocator{RepresentationObjectKind::Module, "top"},
           RepresentationLocator{RepresentationObjectKind::Cell, "top.u0"},
           RepresentationLocator{RepresentationObjectKind::Port, "top.a"},
           RepresentationLocator{RepresentationObjectKind::Pin, "top.u0.a"},
           RepresentationLocator{RepresentationObjectKind::Net, "top.n"},
       })
    if (llvm::Error error = validateRepresentationLocatorSyntax(gate, locator))
      fail(__func__, llvm::toString(std::move(error)));

  for (llvm::StringRef name : {"", ".top", "top.", "top..u", "9top", "top.9u",
                               "top.u[0]", "top/u", "top.\\u"})
    expectError(__func__,
                validateRepresentationLocatorSyntax(
                    rtl, {RepresentationObjectKind::Instance, name.str()}),
                "name");
  expectError(__func__,
              validateRepresentationLocatorSyntax(
                  rtl, {RepresentationObjectKind::Module, "top.child"}),
              "Module");
  expectError(__func__,
              validateRepresentationLocatorSyntax(
                  rtl, {RepresentationObjectKind::Net, "net"}),
              "top-rooted");
  expectError(__func__,
              validateRepresentationLocatorSyntax(
                  gate, {RepresentationObjectKind::Pin, "top.u0"}),
              "terminal");

  for (RepresentationObjectKind kind :
       {RepresentationObjectKind::Cell, RepresentationObjectKind::Pin,
        RepresentationObjectKind::PhysicalObject,
        RepresentationObjectKind::DeviceResource,
        RepresentationObjectKind::Model})
    expectError(__func__,
                validateRepresentationLocatorSyntax(rtl, {kind, "top.object"}),
                "kind");

  for (RepresentationObjectKind kind :
       {RepresentationObjectKind::Instance, RepresentationObjectKind::Register,
        RepresentationObjectKind::Memory,
        RepresentationObjectKind::PhysicalObject,
        RepresentationObjectKind::DeviceResource,
        RepresentationObjectKind::Model})
    expectError(__func__,
                validateRepresentationLocatorSyntax(gate, {kind, "top.object"}),
                "kind");

  const auto model = take(__func__, RepresentationFormatDescriptorRef::get(
                                        RepresentationFormatKind::FabricModel));
  if (llvm::Error error = validateRepresentationLocatorSyntax(
          model, {RepresentationObjectKind::Model,
                  fabricModelRootCanonicalName.str()}))
    fail(__func__, llvm::toString(std::move(error)));
  expectError(__func__,
              validateRepresentationLocatorSyntax(
                  model, {RepresentationObjectKind::Model, "other_model"}),
              "canonical model root");
  expectError(__func__,
              validateRepresentationLocatorSyntax(
                  model, {RepresentationObjectKind::Module,
                          fabricModelRootCanonicalName.str()}),
              "kind");
}

} // namespace

int main() {
  exactBinaryCodecIsClosed();
  exactJsonCodecIsClosed();
  canonicalOrderingUsesEncodedBytes();
  initialFormatsOwnHdlSyntax();
  return EXIT_SUCCESS;
}
