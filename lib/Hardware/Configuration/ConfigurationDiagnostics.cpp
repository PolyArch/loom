#include "Hardware/Configuration/ConfigurationDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

namespace loom::hardware {
namespace {

llvm::StringRef spelling(ConfigurationABIImportVerificationDomain domain) {
  switch (domain) {
  case ConfigurationABIImportVerificationDomain::SourceInvocation:
    return "source_invocation";
  case ConfigurationABIImportVerificationDomain::IndependentReplay:
    return "independent_replay";
  }
  llvm_unreachable("unknown ConfigurationABI import verification domain");
}

} // namespace

void emitPackedConfigurationABIDerivationStatistics(
    const PackedConfigurationABIDerivationStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::ConfigurationAbiDerivation, [&] {
        llvm::json::Object payload;
        payload["construction_count"] = statistics.constructionCount;
        payload["construction_time_ns"] = statistics.constructionNanoseconds;
        payload["source_cache_hits"] = statistics.sourceCacheHits;
        payload["source_cache_misses"] = statistics.sourceCacheMisses;
        payload["relation_cache_hits"] = statistics.relationCacheHits;
        payload["relation_cache_misses"] = statistics.relationCacheMisses;
        payload["retained_cache_bytes"] = statistics.retainedCacheBytes;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["programming_unit_count"] = statistics.programmingUnitCount;
        payload["configuration_field_count"] =
            statistics.configurationFieldCount;
        payload["encoding_relation_count"] = statistics.encodingRelationCount;
        return llvm::json::Value(std::move(payload));
      });
}

void emitConfigurationABIConstructionStatistics(
    const ConfigurationABIConstructionStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::ConfigurationAbiConstruction, [&] {
        llvm::json::Object payload;
        payload["canonicalization_count"] = statistics.canonicalizationCount;
        payload["construction_time_ns"] =
            statistics.canonicalizationNanoseconds;
        payload["cache_hits"] = statistics.semanticValidationCacheHits;
        payload["cache_misses"] = statistics.semanticValidationCacheMisses;
        payload["physical_slot_validation_count"] =
            statistics.physicalSlotValidationCount;
        payload["retained_cache_bytes"] = statistics.retainedCacheBytes;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["encoding_relation_count"] = statistics.encodingRelationCount;
        payload["configuration_field_count"] =
            statistics.configurationFieldCount;
        payload["canonical_byte_count"] = statistics.canonicalByteCount;
        return llvm::json::Value(std::move(payload));
      });
}

void emitConfigurationABIImportSessionStatistics(
    ConfigurationABIImportVerificationDomain domain,
    const ConfigurationABIImportSessionStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::ConfigurationAbiImportSession, [&] {
        llvm::json::Object payload;
        payload["verification_domain"] = spelling(domain);
        payload["import_requests"] = statistics.importRequests;
        payload["unique_constructions"] = statistics.uniqueConstructions;
        payload["cache_hits"] = statistics.cacheHits;
        payload["cache_misses"] = statistics.cacheMisses;
        payload["bytes_read"] = statistics.bytesRead;
        payload["bytes_copied"] = statistics.bytesCopied;
        payload["construction_time_ns"] = statistics.constructionNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["retained_bytes"] = statistics.retainedBytes;
        payload["entry_count"] = statistics.entryCount;
        return llvm::json::Value(std::move(payload));
      });
}

} // namespace loom::hardware
