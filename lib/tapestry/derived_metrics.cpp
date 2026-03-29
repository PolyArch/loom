#include "tapestry/derived_metrics.h"

#include <algorithm>
#include <numeric>

namespace tapestry {

unsigned dataTypeBytes(const std::string &typeName) {
  if (typeName == "f32" || typeName == "i32" || typeName == "u32")
    return 4;
  if (typeName == "f64" || typeName == "i64" || typeName == "u64")
    return 8;
  if (typeName == "f16" || typeName == "i16" || typeName == "u16")
    return 2;
  if (typeName == "i8" || typeName == "u8")
    return 1;
  return 0;
}

DerivedContractMetrics computeDerivedMetrics(const Contract &contract,
                                             unsigned elementSizeBytes) {
  DerivedContractMetrics m;

  // dataVolume from the contract (bytes per invocation), default to 1.
  int64_t volume = static_cast<int64_t>(contract.dataVolume.value_or(1));

  // bandwidth = dataVolume * element_size (bytes/cycle).
  m.bandwidth = static_cast<double>(volume) * elementSizeBytes;

  // dataVolume = volume * element_size (total bytes per tile transfer).
  m.dataVolume = volume * static_cast<int64_t>(elementSizeBytes);

  // crossesCores, NoC bandwidth, and SPM bytes are set later by
  // updatePostAssignment() once core assignment is known.
  m.crossesCores = false;
  m.requiredNoCBandwidth = 0.0;
  m.requiredSPMBytes = 0;

  return m;
}

void updatePostAssignment(DerivedContractMetrics &metrics, bool crossesCores,
                          Placement placement) {
  metrics.crossesCores = crossesCores;

  // requiredNoCBandwidth: bandwidth when crossesCores, else 0.
  metrics.requiredNoCBandwidth = crossesCores ? metrics.bandwidth : 0.0;

  // requiredSPMBytes: dataVolume when visibility==LOCAL_SPM.
  metrics.requiredSPMBytes =
      (placement == Placement::LOCAL_SPM) ? metrics.dataVolume : 0;
}

} // namespace tapestry
