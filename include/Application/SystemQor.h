#ifndef LOOM_APPLICATION_SYSTEMQOR_H
#define LOOM_APPLICATION_SYSTEMQOR_H

#include "Application/RuntimeManifest.h"
#include "Evaluation/Evidence.h"
#include "Simulator/SimulationExecution.h"

namespace llvm::json { class OStream; }
namespace loom { struct ResolvedConfig; }
namespace loom::application {

inline constexpr llvm::StringLiteral applicationSystemQorProjectionSchema =
    "loom.application.system_qor_projection";
inline constexpr llvm::StringLiteral applicationSystemQorProjectionVersion = "1.0";

/// Supported bandwidth qualification path: strictly faster than the same-System
/// source host and strictly above 90 percent of shared-memory acceptance service.
/// Compute occupancy remains unmeasured and cannot supply the other OR branch.
inline constexpr std::uint64_t applicationMinimumMemoryUtilizationNumerator = 9;
inline constexpr std::uint64_t applicationMinimumMemoryUtilizationDenominator = 10;

enum class ApplicationSystemQorStatus : std::uint8_t { Qualified, NotQualified };

struct ApplicationSystemRunEvidence final {
  ArtifactRootReference execution;
  ArtifactRootReference evidence;
  std::optional<ArtifactRootReference> productOracleEvidence;
};

/// Derived measurements remain backed by their exact immutable owner roots.
struct ApplicationSystemRunMeasurement final {
  ApplicationSystemRunEvidence roots;
  ArtifactRootReference request;
  std::optional<ArtifactRootReference> productOracleRequest;
  std::uint64_t elapsedTicks;
  sim::SystemMemoryActivity memoryActivity;
  evaluation::ExactRatio memoryUtilization;
};

/// A validated post-execution relation, not another Artifact family or mutable
/// completion flag. Only qualification can construct it; JSON is a projection.
class ApplicationSystemQor final {
public:
  const ArtifactRootReference &runtimeManifest() const { return manifest_; }
  const ArtifactRootReference &gem5Binding() const { return binding_; }
  const ApplicationSystemRunMeasurement &hostOnly() const { return host_; }
  const ApplicationSystemRunMeasurement &candidate() const { return candidate_; }
  evaluation::ExactRatio speedup() const { return speedup_; }
  ApplicationSystemQorStatus status() const;

private:
  ApplicationSystemQor(ArtifactRootReference manifest, ArtifactRootReference binding,
                       ApplicationSystemRunMeasurement host,
                       ApplicationSystemRunMeasurement candidate,
                       evaluation::ExactRatio speedup)
      : manifest_(std::move(manifest)), binding_(std::move(binding)),
        host_(std::move(host)), candidate_(std::move(candidate)), speedup_(speedup) {}
  ArtifactRootReference manifest_;
  ArtifactRootReference binding_;
  ApplicationSystemRunMeasurement host_;
  ApplicationSystemRunMeasurement candidate_;
  evaluation::ExactRatio speedup_;

  friend llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
      const FinalizedApplicationRuntimeManifest &, const ApplicationSystemRunEvidence &,
      const evaluation::CaseArtifactResolution &, const ApplicationSystemRunEvidence &,
      const evaluation::CaseArtifactResolution &, const ResolvedConfig &,
      const ArtifactStore &, const BlobStore &);
};

llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationSystemRunEvidence &hostOnly,
    const evaluation::CaseArtifactResolution &hostResolution,
    const ApplicationSystemRunEvidence &candidate,
    const evaluation::CaseArtifactResolution &candidateResolution,
    const ResolvedConfig &config, const ArtifactStore &artifacts, const BlobStore &blobs);

void writeApplicationSystemQorJsonFields(llvm::json::OStream &json,
                                       const ApplicationSystemQor &qor);

} // namespace loom::application
#endif
