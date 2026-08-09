#include "Hardware/Implementation/DefPhysical.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

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
    fail(test, "accepted invalid DEF");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::string validDef(llvm::StringRef extraPower = {}) {
  return "VERSION 5.8 ;\n"
         "DESIGN top ;\n"
         "PINS 2 ;\n"
         "- VPWR + NET power_main + DIRECTION INOUT + USE POWER "
         "+ LAYER M4 ( 0 0 ) ( 10 10 ) + FIXED ( 20 20 ) N ;\n"
         "- VGND + NET ground_main + DIRECTION INOUT + USE GROUND "
         "+ LAYER M4 ( 0 0 ) ( 10 10 ) + FIXED ( 40 20 ) N ;\n"
         "END PINS\n"
         "SPECIALNETS " +
         std::to_string(extraPower.empty() ? 2 : 3) +
         " ;\n"
         "- power_main + USE POWER + ROUTED M4 ( 20 20 ) ( 100 20 ) ;\n"
         "- ground_main + USE GROUND + ROUTED M4 ( 40 20 ) ( 100 40 ) ;\n" +
         extraPower.str() +
         "END SPECIALNETS\n"
         "NETS 1 ;\n"
         "- signal_a ( u0 A ) ( u1 Z ) + ROUTED M2 ( 1 1 ) ( 2 2 ) ;\n"
         "END NETS\n"
         "END DESIGN\n";
}

void singleSupplyNetworkIsExact() {
  const DefPhysicalDesign design = take(
      __func__, parseDefPhysicalDesign(validDef(), "top",
                                       RepresentationPhysicalStage::Routed));
  const auto network = deriveDefSingleSupplyNetwork(design);
  require(__func__,
          network && network->powerNet == "power_main" &&
              network->groundNet == "ground_main",
          "single supply projection changed");
}

void incompatibleNetworksAreNotInvented() {
  const DefPhysicalDesign multi = take(
      __func__,
      parseDefPhysicalDesign(validDef("- auxiliary + USE POWER + ROUTED M4 "
                                      "( 0 0 ) ( 1 1 ) ;\n"),
                             "top", RepresentationPhysicalStage::Routed));
  require(__func__, !deriveDefSingleSupplyNetwork(multi),
          "multi-domain DEF was collapsed to one supply");

  std::string partial = validDef();
  const std::string routed =
      "- power_main + USE POWER + ROUTED M4 ( 20 20 ) ( 100 20 ) ;";
  const std::size_t offset = partial.find(routed);
  require(__func__, offset != std::string::npos, "fixture route is absent");
  partial.replace(offset, routed.size(), "- power_main + USE POWER ;");
  const DefPhysicalDesign incomplete =
      take(__func__, parseDefPhysicalDesign(
                         partial, "top", RepresentationPhysicalStage::Routed));
  require(__func__, !deriveDefSingleSupplyNetwork(incomplete),
          "partial supply network was accepted");
}

void malformedDefIsRejected() {
  expectError(__func__,
              parseDefPhysicalDesign(validDef(), "other",
                                     RepresentationPhysicalStage::Routed),
              "exact representation top");
  std::string wrongCount = validDef();
  const std::size_t count = wrongCount.find("PINS 2");
  wrongCount.replace(count, 6, "PINS 3");
  expectError(__func__,
              parseDefPhysicalDesign(wrongCount, "top",
                                     RepresentationPhysicalStage::Routed),
              "entry count");
  expectError(__func__,
              parseDefPhysicalDesign("DESIGN top ;\nEND DESIGN\n", "top",
                                     RepresentationPhysicalStage::Routed),
              "no routed DEF network");
}

} // namespace

int main() {
  singleSupplyNetworkIsExact();
  incompatibleNetworksAreNotInvented();
  malformedDefIsRejected();
  return EXIT_SUCCESS;
}
