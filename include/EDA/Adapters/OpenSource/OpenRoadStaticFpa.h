#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADSTATICFPA_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADSTATICFPA_H

#include "EDA/Adapters/OpenSource/OpenRoad.h"
#include "Evaluation/Models/OpenRoadStaticFpa.h"

namespace loom::eda::open_source {

struct OpenRoadStaticFpaObservation final {
  std::optional<evaluation::DecimalValue> limitingClockFrequencyHertz;
  std::optional<evaluation::DecimalValue> totalAreaSquareMeters;
  std::optional<evaluation::DecimalValue> dynamicPowerWatts;
  std::optional<evaluation::DecimalValue> leakagePowerWatts;

  friend bool operator==(const OpenRoadStaticFpaObservation &lhs,
                         const OpenRoadStaticFpaObservation &rhs) {
    return lhs.limitingClockFrequencyHertz == rhs.limitingClockFrequencyHertz &&
           lhs.totalAreaSquareMeters == rhs.totalAreaSquareMeters &&
           lhs.dynamicPowerWatts == rhs.dynamicPowerWatts &&
           lhs.leakagePowerWatts == rhs.leakagePowerWatts;
  }
};

struct OpenRoadStaticFpaDriverFiles final {
  std::vector<std::string> netlists;
  std::vector<std::string> constraints;
  std::string physicalDatabase;
  std::string technologyLef;
  std::string cellLef;
  std::string liberty;
};

struct OpenRoadStaticFpaDriverConfiguration final {
  std::string top;
  OpenRoadStaticFpaDriverFiles files;
  evaluation::models::CompleteOpenRoadStaticFpaConfiguration analysis;
};

llvm::Expected<std::string> renderOpenRoadStaticFpaDriver(
    const OpenRoadStaticFpaDriverConfiguration &configuration);

llvm::Expected<std::string>
renderOpenRoadStaticFpaPublisher(llvm::ArrayRef<evaluation::MetricKind> metrics);

llvm::Expected<OpenRoadStaticFpaObservation>
parseOpenRoadStaticFpaResult(llvm::StringRef contents,
                             llvm::StringRef expectedTop,
                             llvm::ArrayRef<evaluation::MetricKind> metrics);

llvm::Error registerOpenRoadStaticFpaEvaluationProvider();

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADSTATICFPA_H
