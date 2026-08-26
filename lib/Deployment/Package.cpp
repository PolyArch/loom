#include "Deployment/Package.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(__linux__)
#error "Deployment package publication currently requires Linux"
#endif

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <linux/fs.h>
#include <string>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace loom::deployment {
namespace {

using MonotonicClock = std::chrono::steady_clock;

llvm::StringRef spelling(DeploymentPackageOperation operation) {
  switch (operation) {
  case DeploymentPackageOperation::SourceClosure:
    return "source_closure";
  case DeploymentPackageOperation::StagingWrite:
    return "staging_write";
  case DeploymentPackageOperation::IndependentRootImport:
    return "independent_root_import";
  case DeploymentPackageOperation::IndependentClosure:
    return "independent_closure";
  case DeploymentPackageOperation::StagingEntryValidation:
    return "staging_entry_validation";
  case DeploymentPackageOperation::AtomicPublish:
    return "atomic_publish";
  }
  llvm_unreachable("unknown Deployment package operation");
}

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             MonotonicClock::now() - begin)
      .count();
}

DeploymentPackageOperationStatistics operationStatistics(
    DeploymentPackageOperation operation, MonotonicClock::time_point begin,
    const fabric::FabricArtifactImportSessionStatistics &before,
    const fabric::FabricArtifactImportSessionStatistics &after,
    std::uint64_t artifactCount, std::uint64_t blobCount) {
  return {operation,
          elapsedNanoseconds(begin),
          artifactCount,
          blobCount,
          after.cacheHits - before.cacheHits,
          after.cacheMisses - before.cacheMisses,
          after.constructionNanoseconds - before.constructionNanoseconds,
          after.deterministicWork - before.deterministicWork,
          after.retainedPayloadBytes};
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deployment_package_invalid: " + message);
}

llvm::Error ioError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deployment_package_io: " + message);
}

struct DerivedPackageClosure final {
  std::vector<ArtifactRootReference> artifacts;
  std::vector<BlobDigest> blobs;
};

bool blobLess(const BlobDigest &lhs, const BlobDigest &rhs) {
  return lhs.bytes() < rhs.bytes();
}

class ClosureBuilder final {
public:
  ClosureBuilder(const ArtifactStore &artifacts, const BlobStore &blobs)
      : artifacts_(artifacts), blobs_(blobs) {}

  llvm::Expected<DerivedPackageClosure>
  derive(const FinalizedDeployment &deployment) {
    if (llvm::Error error = addArtifactLeaf(deployment.reference()))
      return error;

    const Deployment &root = deployment.deployment();
    if (llvm::Error error = addSystemMapping(root.systemMapping()))
      return error;
    if (llvm::Error error =
            addCompilerTarget(root.hostProgram().compilerTargetBinding()))
      return error;
    if (llvm::Error error = addBlob(root.hostProgram().programBlob()))
      return error;
    for (const ArtifactRootReference &reference :
         root.instructionCoreBinaries())
      if (llvm::Error error = addInstructionBinary(reference))
        return error;
    for (const DeploymentHardwareBinding &binding : root.hardwareBindings()) {
      if (llvm::Error error =
              addHardwareImplementation(binding.hardwareImplementation))
        return error;
      if (llvm::Error error = addRuntimeBinding(binding.runtimePlatformBinding))
        return error;
    }
    for (const ArtifactRootReference &reference : root.configurationImages())
      if (llvm::Error error = addConfigurationImage(reference))
        return error;
    for (const StaticMemoryImageLeaf &memory : root.staticMemoryImages()) {
      if (llvm::Error error = addArtifactLeaf(memory.canonicalDataflow()))
        return error;
      if (llvm::Error error = addCompilerTarget(memory.layoutBinding()))
        return error;
      for (const StaticMemoryInitializedChunk &chunk :
           memory.initializedChunks())
        if (llvm::Error error = addBlob(chunk.blobDigest))
          return error;
    }

    llvm::sort(artifactsInClosure_, artifactRootReferenceLess);
    llvm::sort(blobsInClosure_, blobLess);
    return DerivedPackageClosure{std::move(artifactsInClosure_),
                                 std::move(blobsInClosure_)};
  }

private:
  llvm::Expected<bool>
  addArtifact(const ArtifactRootReference &reference) {
    auto exact = llvm::find(artifactsInClosure_, reference);
    if (exact != artifactsInClosure_.end())
      return false;
    auto colliding = llvm::find_if(
        artifactsInClosure_, [&](const ArtifactRootReference &existing) {
          return existing.artifact == reference.artifact;
        });
    if (colliding != artifactsInClosure_.end())
      return invalid("one ArtifactIdentity is framed by two schema roots");
    auto object = artifacts_.getStoredObject(reference);
    if (!object)
      return object.takeError();
    artifactsInClosure_.push_back(reference);
    return true;
  }

  llvm::Error addArtifactLeaf(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    return llvm::Error::success();
  }

  llvm::Error addBlob(const BlobDigest &digest) {
    if (llvm::is_contained(blobsInClosure_, digest))
      return llvm::Error::success();
    auto bytes = blobs_.get(digest);
    if (!bytes)
      return bytes.takeError();
    blobsInClosure_.push_back(digest);
    return llvm::Error::success();
  }

  llvm::Error addFabric(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto fabric = fabric::importEntireFabricRoot(reference, artifacts_);
    if (!fabric)
      return fabric.takeError();
    for (const fabric::FabricDirectDependency &dependency :
         fabric->directDependencies())
      if (llvm::Error error = addFabric(dependency.root))
        return error;
    return llvm::Error::success();
  }

  llvm::Error addSystemMapping(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto system = mapping::importSystemMapping(reference, artifacts_);
    if (!system)
      return system.takeError();
    const ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        system->view().dataflowIdentity()};
    const ArtifactRootReference fabricReference{
        fabric::fabricArtifactSchema.identity.str(),
        fabric::fabricArtifactSchema.version, system->view().fabricIdentity()};
    if (llvm::Error error = addArtifactLeaf(dataflowReference))
      return error;
    if (llvm::Error error = addFabric(fabricReference))
      return error;
    for (const ArtifactRootReference &spatialReference :
         system->view().executionBindings().spatialMappingImports()) {
      auto spatialAdded = addArtifact(spatialReference);
      if (!spatialAdded)
        return spatialAdded.takeError();
      if (!*spatialAdded)
        continue;
      auto spatial =
          mapping::importSpatialMapping(spatialReference, artifacts_);
      if (!spatial)
        return spatial.takeError();
      const ArtifactRootReference techReference{
          mapping::mappingArtifactSchema.identity.str(),
          mapping::mappingArtifactSchema.version,
          spatial->view().techMappingIdentity()};
      auto techAdded = addArtifact(techReference);
      if (!techAdded)
        return techAdded.takeError();
      if (!*techAdded)
        continue;
      auto tech = mapping::importTechMapping(techReference, artifacts_);
      if (!tech)
        return tech.takeError();
    }
    return llvm::Error::success();
  }

  llvm::Error addCompilerTarget(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto target = importCompilerTargetBinding(reference, artifacts_);
    if (!target)
      return target.takeError();
    const ArtifactRootReference fabricReference{
        fabric::fabricArtifactSchema.identity.str(),
        fabric::fabricArtifactSchema.version,
        target->binding().processorArchitecture().fabricArtifact()};
    if (llvm::Error error = addFabric(fabricReference))
      return error;
    for (const CompilerSupportComponent &component :
         target->binding().supportComponents())
      if (llvm::Error error = addBlob(component.contentBlob))
        return error;
    return llvm::Error::success();
  }

  llvm::Error addInstructionBinary(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto binary = importInstructionCoreBinary(reference, artifacts_, blobs_);
    if (!binary)
      return binary.takeError();
    if (llvm::Error error =
            addArtifactLeaf(binary->binary().canonicalDataflow()))
      return error;
    if (llvm::Error error =
            addCompilerTarget(binary->binary().compilerTargetBinding()))
      return error;
    return addBlob(binary->binary().codeBlob());
  }

  llvm::Error addConfigurationAbi(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto abi = hardware::importConfigurationABI(reference, artifacts_);
    if (!abi)
      return abi.takeError();
    return addFabric(abi->abi().fabric());
  }

  llvm::Error
  addHardwareImplementation(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto implementation =
        hardware::importHardwareImplementation(reference, artifacts_, blobs_);
    if (!implementation)
      return implementation.takeError();
    const hardware::HardwareImplementation &value =
        implementation->implementation();
    if (llvm::Error error = addFabric(value.fabric()))
      return error;
    if (llvm::Error error = addConfigurationAbi(value.configurationAbi()))
      return error;
    if (value.implementationPlatform()) {
      auto platformAdded = addArtifact(*value.implementationPlatform());
      if (!platformAdded)
        return platformAdded.takeError();
      if (*platformAdded) {
        auto platform = platform::importImplementationPlatform(
            *value.implementationPlatform(), artifacts_);
        if (!platform)
          return platform.takeError();
      }
    }
    for (const hardware::ImplementationPayload &payload :
         value.representationRoot().payloads)
      if (llvm::Error error = addBlob(payload.blobDigest))
        return error;
    return llvm::Error::success();
  }

  llvm::Error addRuntimeBinding(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto binding =
        runtime::importRuntimePlatformBinding(reference, artifacts_, blobs_);
    if (!binding)
      return binding.takeError();
    if (llvm::Error error = addHardwareImplementation(
            binding->binding().hardwareImplementation()))
      return error;
    if (const auto *trusted = std::get_if<runtime::TrustedImmutableIdentity>(
            &binding->binding().identityVerification()))
      return addBlob(trusted->attestationBlob);
    return llvm::Error::success();
  }

  llvm::Error addConfigurationImage(const ArtifactRootReference &reference) {
    auto added = addArtifact(reference);
    if (!added)
      return added.takeError();
    if (!*added)
      return llvm::Error::success();
    auto image = importHardwareConfigurationImage(reference, artifacts_);
    if (!image)
      return image.takeError();
    if (llvm::Error error =
            addConfigurationAbi(image->image().configurationAbi()))
      return error;
    return addArtifactLeaf(image->image().sourceMapping().mapping);
  }

  const ArtifactStore &artifacts_;
  const BlobStore &blobs_;
  std::vector<ArtifactRootReference> artifactsInClosure_;
  std::vector<BlobDigest> blobsInClosure_;
};

llvm::Error writeFile(llvm::StringRef path,
                      llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return ioError("cannot create package file: " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error())
    return ioError("cannot write package file");
  return llvm::Error::success();
}

llvm::Error writeFile(llvm::StringRef path, llvm::StringRef text) {
  return writeFile(
      path,
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(text.data()), text.size()));
}

std::string childPath(llvm::StringRef parent, llvm::StringRef child) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, child);
  return path.str().str();
}

llvm::Expected<std::vector<std::string>>
regularEntryNames(llvm::StringRef directory) {
  std::error_code error;
  std::vector<std::string> names;
  for (llvm::sys::fs::directory_iterator iterator(directory, error), end;
       iterator != end && !error; iterator.increment(error)) {
    auto status = iterator->status();
    if (!status)
      return ioError("cannot inspect staging package entry: " +
                     status.getError().message());
    if (!llvm::sys::fs::is_regular_file(*status))
      return invalid("staging package contains a non-regular entry");
    names.push_back(llvm::sys::path::filename(iterator->path()).str());
  }
  if (error)
    return ioError("cannot enumerate staging package: " + error.message());
  llvm::sort(names);
  return names;
}

llvm::Error validateNames(llvm::StringRef directory,
                          std::vector<std::string> expected) {
  auto actual = regularEntryNames(directory);
  if (!actual)
    return actual.takeError();
  llvm::sort(expected);
  if (*actual != expected)
    return invalid("staging package has missing or unreferenced entries");
  return llvm::Error::success();
}

llvm::Error validateStaging(llvm::StringRef staging,
                            const ArtifactRootReference &root,
                            const DeploymentPackageClosure &expected) {
  fabric::FabricArtifactImportSession importSession(
      fabric::FabricArtifactImportSessionMode::Isolated);
  hardware::ConfigurationABIImportSession configurationAbiImportSession(
      hardware::ConfigurationABIImportSessionMode::Isolated);
  auto before = importSession.statistics();
  auto begin = MonotonicClock::now();
  const std::string objects = childPath(staging, "objects");
  const std::string blobs = childPath(staging, "blobs");
  ArtifactStore artifactStore(objects);
  BlobStore blobStore(blobs);
  mapping::SystemMappingImportSession systemMappingImportSession(artifactStore,
                                                                 64);
  ConfigurationImageProjectionSession projectionSession(artifactStore, 1);
  auto deployment = importDeployment(root, artifactStore, blobStore);
  if (!deployment)
    return invalid("staging package cannot import its root: " +
                   llvm::toString(deployment.takeError()));
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::IndependentRootImport, begin, before,
      importSession.statistics(), expected.artifacts().size(),
      expected.blobs().size()));

  before = importSession.statistics();
  begin = MonotonicClock::now();
  ClosureBuilder builder(artifactStore, blobStore);
  auto actual = deriveDeploymentPackageClosure(*deployment, artifactStore,
                                                blobStore);
  if (!actual)
    return actual.takeError();
  if (actual->artifacts() != expected.artifacts() ||
      actual->blobs() != expected.blobs())
    return invalid("staging package closure differs after empty-store import");
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::IndependentClosure, begin, before,
      importSession.statistics(), actual->artifacts().size(),
      actual->blobs().size()));

  begin = MonotonicClock::now();
  std::vector<std::string> artifactNames;
  artifactNames.reserve(expected.artifacts().size());
  for (const ArtifactRootReference &reference : expected.artifacts())
    artifactNames.push_back(formatArtifactIdentityHex(reference.artifact));
  if (llvm::Error error = validateNames(objects, std::move(artifactNames)))
    return error;
  std::vector<std::string> blobNames;
  blobNames.reserve(expected.blobs().size());
  for (const BlobDigest &digest : expected.blobs())
    blobNames.push_back(formatBlobDigestHex(digest));
  if (llvm::Error error = validateNames(blobs, std::move(blobNames)))
    return error;
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::StagingEntryValidation, begin,
      importSession.statistics(), importSession.statistics(),
      expected.artifacts().size(), expected.blobs().size()));
  hardware::emitConfigurationABIImportSessionStatistics(
      hardware::ConfigurationABIImportVerificationDomain::IndependentReplay,
      configurationAbiImportSession.statistics());
  emitConfigurationImageProjectionSessionStatistics(
      ConfigurationImageProjectionVerificationDomain::IndependentReplay,
      projectionSession.statistics());
  mapping::emitSystemMappingImportSessionStatistics(
      mapping::SystemMappingImportVerificationDomain::IndependentReplay,
      systemMappingImportSession.statistics());
  return llvm::Error::success();
}

llvm::Error publishNoReplace(llvm::StringRef staging,
                             llvm::StringRef destination) {
  const std::string source = staging.str();
  const std::string target = destination.str();
  int result;
  do {
    result =
        static_cast<int>(::syscall(SYS_renameat2, AT_FDCWD, source.c_str(),
                                   AT_FDCWD, target.c_str(), RENAME_NOREPLACE));
  } while (result == -1 && errno == EINTR);
  if (result == 0)
    return llvm::Error::success();
  if (errno == EEXIST)
    return invalid("deployment package output already exists");
  return ioError("cannot atomically publish deployment package: " +
                 llvm::errnoAsErrorCode().message());
}

} // namespace

void emitDeploymentPackageOperationStatistics(
    const DeploymentPackageOperationStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::DeploymentPackageStatistics, [&] {
        llvm::json::Object payload;
        payload["operation"] = spelling(statistics.operation);
        payload["duration_ns"] = statistics.durationNanoseconds;
        payload["artifact_count"] = statistics.artifactCount;
        payload["blob_count"] = statistics.blobCount;
        payload["fabric_import_cache_hits"] =
            statistics.fabricImportCacheHits;
        payload["fabric_import_cache_misses"] =
            statistics.fabricImportCacheMisses;
        payload["fabric_import_construction_time_ns"] =
            statistics.fabricImportConstructionNanoseconds;
        payload["fabric_import_deterministic_work"] =
            statistics.fabricImportDeterministicWork;
        payload["fabric_import_retained_payload_bytes"] =
            statistics.fabricImportRetainedPayloadBytes;
        return llvm::json::Value(std::move(payload));
      });
}

llvm::Expected<DeploymentPackageClosure>
deriveDeploymentPackageClosure(const FinalizedDeployment &deployment,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  ClosureBuilder builder(artifacts, blobs);
  auto closure = builder.derive(deployment);
  if (!closure)
    return closure.takeError();
  return DeploymentPackageClosure(std::move(closure->artifacts),
                                  std::move(closure->blobs));
}

llvm::Error publishDeploymentPackage(const FinalizedDeployment &deployment,
                                     llvm::StringRef outputPath,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs) {
  fabric::FabricArtifactImportSession importSession;
  hardware::ConfigurationABIImportSession configurationAbiImportSession;
  if (outputPath.empty())
    return invalid("deployment package output path is empty");
  auto before = importSession.statistics();
  auto begin = MonotonicClock::now();
  auto closure =
      deriveDeploymentPackageClosure(deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::SourceClosure, begin, before,
      importSession.statistics(), closure->artifacts().size(),
      closure->blobs().size()));

  llvm::SmallString<256> destination(outputPath);
  llvm::sys::path::remove_dots(destination, true);
  const llvm::StringRef filename = llvm::sys::path::filename(destination);
  if (filename.empty() || filename == "." || filename == "..")
    return invalid("deployment package output has no directory name");
  llvm::SmallString<256> parent = llvm::sys::path::parent_path(destination);
  if (parent.empty())
    parent = ".";
  llvm::SmallString<256> stagingModel(parent);
  llvm::sys::path::append(stagingModel,
                          ("." + filename + ".loom-package").str());
  llvm::SmallString<256> staging;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingModel, staging))
    return ioError("cannot create staging package: " + error.message());
  bool stagingExists = true;
  llvm::scope_exit cleanup([&] {
    if (stagingExists)
      std::filesystem::remove_all(staging.str().str());
  });

  before = importSession.statistics();
  begin = MonotonicClock::now();
  const std::string objects = childPath(staging, "objects");
  const std::string blobDirectory = childPath(staging, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directory(objects))
    return ioError("cannot create package objects directory: " +
                   error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(blobDirectory))
    return ioError("cannot create package blobs directory: " + error.message());

  const std::string rootText =
      formatArtifactIdentityHex(deployment.reference().artifact);
  if (llvm::Error error = writeFile(childPath(staging, "root"), rootText))
    return error;
  for (const ArtifactRootReference &reference : closure->artifacts()) {
    auto bytes = artifacts.getStoredObject(reference);
    if (!bytes)
      return bytes.takeError();
    if (llvm::Error error = writeFile(
            childPath(objects, formatArtifactIdentityHex(reference.artifact)),
            *bytes))
      return error;
  }
  for (const BlobDigest &digest : closure->blobs()) {
    auto bytes = blobs.get(digest);
    if (!bytes)
      return bytes.takeError();
    if (llvm::Error error = writeFile(
            childPath(blobDirectory, formatBlobDigestHex(digest)), *bytes))
      return error;
  }
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::StagingWrite, begin, before,
      importSession.statistics(), closure->artifacts().size(),
      closure->blobs().size()));
  if (llvm::Error error =
          validateStaging(staging, deployment.reference(), *closure))
    return error;
  begin = MonotonicClock::now();
  if (llvm::Error error = publishNoReplace(staging, destination))
    return error;
  emitDeploymentPackageOperationStatistics(operationStatistics(
      DeploymentPackageOperation::AtomicPublish, begin,
      importSession.statistics(), importSession.statistics(),
      closure->artifacts().size(), closure->blobs().size()));
  hardware::emitConfigurationABIImportSessionStatistics(
      hardware::ConfigurationABIImportVerificationDomain::SourceInvocation,
      configurationAbiImportSession.statistics());
  stagingExists = false;
  return llvm::Error::success();
}

} // namespace loom::deployment
