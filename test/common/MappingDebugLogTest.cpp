#include "Common/MappingDebugLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "mapping debug log test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

} // namespace

int main(int argc, char **argv) {
  using namespace loom::mapping_debug;
  if (argc != 2)
    fail("expected one level argument");

  std::uint64_t expected = 0;
  if (llvm::StringRef(argv[1]).getAsInteger(10, expected) || expected > 3)
    fail("invalid expected level");
  if (static_cast<std::uint64_t>(level()) != expected)
    fail("environment level was not parsed once into the closed domain");

  std::array<bool, 3> built = {false, false, false};
  emit(Level::Summary, Stage::TechMapping, Event::InvocationBegin,
       [&](llvm::json::Object &fields) {
         built[0] = true;
         fields["candidate"] = 7;
       });
  emit(Level::Decision, Stage::SpatialPnr, Event::NegotiationIteration,
       [&](llvm::json::Object &fields) {
         built[1] = true;
         fields["iteration"] = 11;
       });
  emit(Level::Detail, Stage::SystemPnr, Event::CapacityConflict,
       [&](llvm::json::Object &fields) {
         built[2] = true;
         fields["usage"] = 2;
         fields["capacity"] = 1;
       });

  for (std::size_t index = 0; index != built.size(); ++index)
    if (built[index] != (index < expected))
      fail("disabled event constructed diagnostic fields");

  MappingRunStatistics statistics;
  statistics.candidateRows = 13;
  statistics.candidatePublications = 5;
  statistics.actionsProposed = 17;
  statistics.actionsAccepted = 8;
  statistics.actionsRejected = 6;
  statistics.actionsRolledBack = 3;
  statistics.aStarExpansions = 101;
  statistics.negotiatedIterations = 11;
  statistics.capacityConflicts = 4;
  statistics.arithmeticFailures = 1;
  statistics.emit(Stage::SystemPnr, ClosureStatus::Closed);
  return EXIT_SUCCESS;
}
