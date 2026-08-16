#include "Hardware/Configuration/ConfigurationDiagnostics.h"

int main() {
  loom::hardware::PackedConfigurationABIDerivationStatistics derivation;
  derivation.constructionCount = 1;
  derivation.constructionNanoseconds = 2;
  derivation.sourceCacheHits = 3;
  derivation.sourceCacheMisses = 4;
  derivation.relationCacheHits = 5;
  derivation.relationCacheMisses = 6;
  derivation.retainedCacheBytes = 7;
  derivation.deterministicWork = 8;
  derivation.programmingUnitCount = 9;
  derivation.configurationFieldCount = 10;
  derivation.encodingRelationCount = 11;
  loom::hardware::emitPackedConfigurationABIDerivationStatistics(derivation);

  loom::hardware::ConfigurationABIConstructionStatistics construction;
  construction.canonicalizationCount = 12;
  construction.canonicalizationNanoseconds = 13;
  construction.semanticValidationCacheHits = 14;
  construction.semanticValidationCacheMisses = 15;
  construction.physicalSlotValidationCount = 16;
  construction.retainedCacheBytes = 17;
  construction.deterministicWork = 18;
  construction.encodingRelationCount = 19;
  construction.configurationFieldCount = 20;
  construction.canonicalByteCount = 21;
  loom::hardware::emitConfigurationABIConstructionStatistics(construction);

  loom::hardware::ConfigurationABIImportSessionStatistics importSession;
  importSession.importRequests = 22;
  importSession.uniqueConstructions = 23;
  importSession.cacheHits = 24;
  importSession.cacheMisses = 25;
  importSession.bytesRead = 26;
  importSession.bytesCopied = 27;
  importSession.constructionNanoseconds = 28;
  importSession.deterministicWork = 29;
  importSession.retainedBytes = 30;
  importSession.entryCount = 31;
  loom::hardware::emitConfigurationABIImportSessionStatistics(
      loom::hardware::ConfigurationABIImportVerificationDomain::SourceInvocation,
      importSession);
  return 0;
}
