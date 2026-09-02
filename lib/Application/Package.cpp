#include "Application/Package.h"

#include "Application/ActivationDecision.h"
#include "Application/Build.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Deployment/Package.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(__linux__)
#error "Application package publication currently requires Linux"
#endif

#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <linux/fs.h>
#include <set>
#include <string>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

struct ApplicationPackageClosure final {
  std::vector<ArtifactRootReference> artifacts;
  std::vector<BlobDigest> blobs;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "application_package_invalid: " + message);
}

llvm::Error ioError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "application_package_io: " + message);
}

bool blobLess(const BlobDigest &lhs, const BlobDigest &rhs) {
  return lhs.bytes() < rhs.bytes();
}

std::string childPath(llvm::StringRef parent, llvm::StringRef name) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, name);
  return path.str().str();
}

llvm::Error writeFile(llvm::StringRef path,
                      llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return ioError("cannot open package file '" + path +
                   "': " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error())
    return ioError("cannot write package file '" + path + "'");
  return llvm::Error::success();
}

llvm::Error writeText(llvm::StringRef path, llvm::StringRef text) {
  return writeFile(
      path, {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()});
}

llvm::Expected<std::string> readText(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return ioError("cannot read package file '" + path +
                   "': " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

llvm::Expected<ArtifactIdentity> readIdentity(llvm::StringRef path,
                                              llvm::StringRef name) {
  auto text = readText(path);
  if (!text)
    return text.takeError();
  if (text->size() != 64)
    return invalid(name + " is not one SHA-256 identity");
  auto identity = parseArtifactIdentityHex(*text);
  if (!identity)
    return invalid(name +
                   " is malformed: " + llvm::toString(identity.takeError()));
  return *identity;
}

llvm::Error addArtifact(std::vector<ArtifactRootReference> &roots,
                        const ArtifactRootReference &root,
                        const ArtifactStore &artifacts) {
  if (llvm::is_contained(roots, root))
    return llvm::Error::success();
  if (llvm::any_of(roots, [&](const auto &existing) {
        return existing.artifact == root.artifact;
      }))
    return invalid("one Artifact identity has multiple schema framings");
  auto bytes = artifacts.getStoredObject(root);
  if (!bytes)
    return bytes.takeError();
  roots.push_back(root);
  return llvm::Error::success();
}

void addBlob(std::vector<BlobDigest> &digests, const BlobDigest &digest) {
  if (!llvm::is_contained(digests, digest))
    digests.push_back(digest);
}

llvm::Error addRequestPayloadBlobs(
    std::vector<BlobDigest> &blobs,
    llvm::ArrayRef<ArtifactRootReference> requestDependencies,
    const ArtifactStore &artifacts) {
  for (const ArtifactRootReference &root : requestDependencies) {
    if (root.schemaIdentity !=
            evaluation::modelParameterBundleSchema.identity ||
        root.schemaVersion != evaluation::modelParameterBundleSchema.version)
      continue;
    auto bundle = evaluation::importModelParameterBundleRoot(root, artifacts);
    if (!bundle)
      return invalid("model parameter bundle failed strict root import: " +
                     llvm::toString(bundle.takeError()));
    addBlob(blobs, bundle->payloadDigest());
  }
  return llvm::Error::success();
}

llvm::Error addFabricClosure(std::vector<ArtifactRootReference> &roots,
                             std::vector<ArtifactRootReference> &expanded,
                             const ArtifactRootReference &root,
                             const ArtifactStore &artifacts) {
  if (llvm::is_contained(expanded, root))
    return llvm::Error::success();
  if (llvm::Error error = addArtifact(roots, root, artifacts))
    return error;
  auto imported = fabric::importEntireFabricRoot(root, artifacts);
  if (!imported)
    return imported.takeError();
  expanded.push_back(root);
  for (const fabric::FabricDirectDependency &dependency :
       imported->directDependencies())
    if (llvm::Error error =
            addFabricClosure(roots, expanded, dependency.root, artifacts))
      return error;
  return llvm::Error::success();
}

llvm::Expected<ApplicationPackageClosure> deriveApplicationPackageClosure(
    const FinalizedApplicationRuntimeManifest &manifest,
    const deployment::FinalizedDeployment &entryDeployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (manifest.manifest().deployment() != entryDeployment.reference())
    return llvm::make_error<ApplicationRuntimeManifestError>(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "runtime manifest does not bind the package entry Deployment");
  ApplicationPackageClosure result;
  std::vector<ArtifactRootReference> deployments = {
      entryDeployment.reference()};
  if (manifest.manifest().transitionGraph())
    for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
         manifest.manifest().transitionGraph()->endpoints)
      if (endpoint.deployment &&
          !llvm::is_contained(deployments, *endpoint.deployment))
        deployments.push_back(*endpoint.deployment);
  for (const ArtifactRootReference &reference : deployments) {
    auto imported = deployment::importDeployment(reference, artifacts, blobs);
    if (!imported)
      return imported.takeError();
    auto closure =
        deployment::deriveDeploymentPackageClosure(*imported, artifacts, blobs);
    if (!closure)
      return closure.takeError();
    for (const ArtifactRootReference &root : closure->artifacts())
      if (llvm::Error error = addArtifact(result.artifacts, root, artifacts))
        return std::move(error);
    for (const BlobDigest &digest : closure->blobs())
      addBlob(result.blobs, digest);
  }

  const ApplicationRuntimeManifest &runtime = manifest.manifest();
  for (const ArtifactRootReference *root :
       {&manifest.reference(), &runtime.sourceProgram(), &runtime.fabric(),
        &runtime.workload(), &runtime.runtimeInput(), &runtime.selectedSystem(),
        &runtime.selectedMapping(), &runtime.deployment(),
        &runtime.activationWorkload(), &runtime.activationRuntimeInput(),
        &runtime.activationDecision()})
    if (llvm::Error error = addArtifact(result.artifacts, *root, artifacts))
      return std::move(error);
  auto activation = importApplicationActivationDecision(
      runtime.activationDecision(), artifacts, blobs);
  if (!activation)
    return activation.takeError();
  auto activationDependencies =
      projectApplicationActivationDecisionDependencies(activation->decision(),
                                                       artifacts, blobs);
  if (!activationDependencies)
    return activationDependencies.takeError();
  for (const ArtifactRootReference &root : activationDependencies->artifacts)
    if (llvm::Error error = addArtifact(result.artifacts, root, artifacts))
      return std::move(error);
  for (const BlobDigest &digest : activationDependencies->blobs)
    addBlob(result.blobs, digest);
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       runtime.sourceBackedReplayCases())
    for (const ArtifactRootReference *root :
         {&replay.workload, &replay.runtimeInput})
      if (llvm::Error error = addArtifact(result.artifacts, *root, artifacts))
        return std::move(error);
  std::vector<ArtifactRootReference> expandedFabrics;
  if (llvm::Error error = addFabricClosure(result.artifacts, expandedFabrics,
                                           runtime.fabric(), artifacts))
    return std::move(error);
  if (llvm::Error error = addFabricClosure(result.artifacts, expandedFabrics,
                                           runtime.selectedSystem(), artifacts))
    return std::move(error);
  if (runtime.transitionGraph())
    for (const pnr::ResourceTimeTransition &transition :
         runtime.transitionGraph()->transitions) {
      if (transition.safePoint)
        if (llvm::Error error = addArtifact(
                result.artifacts, transition.safePoint->artifact, artifacts))
          return std::move(error);
    }
  for (const ArtifactRootReference &root : runtime.runtimeRequestDependencies())
    if (llvm::Error error = addArtifact(result.artifacts, root, artifacts))
      return std::move(error);

  for (const ArtifactRootReference &evidence : runtime.runtimeEvidence()) {
    if (llvm::Error error = addArtifact(result.artifacts, evidence, artifacts))
      return std::move(error);
    auto projection = evaluation::importEvaluationEvidenceDependencyProjection(
        evidence, artifacts);
    if (!projection)
      return projection.takeError();
    if (llvm::Error error =
            addArtifact(result.artifacts, projection->request, artifacts))
      return std::move(error);
    auto requestDependencies =
        evaluation::importEvaluationRequestArtifactReferences(
            projection->request, artifacts);
    if (!requestDependencies)
      return requestDependencies.takeError();
    for (const ArtifactRootReference &root : *requestDependencies)
      if (llvm::Error error = addArtifact(result.artifacts, root, artifacts))
        return std::move(error);
    if (llvm::Error error = addRequestPayloadBlobs(
            result.blobs, *requestDependencies, artifacts))
      return std::move(error);
    for (const evaluation::ModelOutputBinding &binding :
         projection->outputBindings)
      for (const ArtifactRootReference &root : binding.artifacts)
        if (llvm::Error error = addArtifact(result.artifacts, root, artifacts))
          return std::move(error);
  }
  llvm::sort(result.artifacts, artifactRootReferenceLess);
  llvm::sort(result.blobs, blobLess);
  return result;
}

llvm::Expected<std::set<std::string>>
regularEntryNames(llvm::StringRef directory) {
  std::error_code error;
  std::set<std::string> names;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(directory.str(), error)) {
    if (error || entry.is_symlink(error) || !entry.is_regular_file(error))
      return invalid("package store contains a non-regular entry");
    names.insert(entry.path().filename().string());
  }
  if (error)
    return ioError("cannot enumerate package store: " + error.message());
  return names;
}

llvm::Error validateStoreEntries(llvm::StringRef directory,
                                 const std::set<std::string> &expected) {
  auto actual = regularEntryNames(directory);
  if (!actual)
    return actual.takeError();
  if (*actual != expected)
    return invalid("package has missing or unreferenced store entries");
  return llvm::Error::success();
}

llvm::Error validateTopLevel(llvm::StringRef packagePath) {
  const std::set<std::string> expected = {"application", "blobs", "objects",
                                          "root"};
  std::error_code error;
  std::set<std::string> actual;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(packagePath.str(), error)) {
    if (error || entry.is_symlink(error))
      return invalid("package top level contains an invalid entry");
    const std::string name = entry.path().filename().string();
    if ((name == "objects" || name == "blobs") && !entry.is_directory(error))
      return invalid("package store entry is not a directory");
    if ((name == "root" || name == "application") &&
        !entry.is_regular_file(error))
      return invalid("package root entry is not a regular file");
    actual.insert(name);
  }
  if (error)
    return ioError("cannot enumerate package top level: " + error.message());
  if (actual != expected)
    return invalid("package has missing or unreferenced top-level entries");
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
    return invalid("application package output already exists");
  return ioError("cannot atomically publish application package: " +
                 llvm::errnoAsErrorCode().message());
}

} // namespace

llvm::Expected<ImportedApplicationPackage>
importApplicationPackage(llvm::StringRef packagePath) {
  if (llvm::Error error = validateTopLevel(packagePath))
    return std::move(error);
  const ArtifactStore artifacts(childPath(packagePath, "objects"));
  const BlobStore blobs(childPath(packagePath, "blobs"));
  fabric::FabricArtifactImportSession fabricImportSession(
      fabric::FabricArtifactImportSessionMode::Isolated);
  hardware::ConfigurationABIImportSession configurationAbiImportSession(
      hardware::ConfigurationABIImportSessionMode::Isolated);
  mapping::SystemMappingImportSession systemMappingImportSession(artifacts, 64);
  deployment::ConfigurationImageProjectionSession projectionSession(artifacts,
                                                                    64);
  auto deploymentIdentity =
      readIdentity(childPath(packagePath, "root"), "Deployment package root");
  if (!deploymentIdentity)
    return deploymentIdentity.takeError();
  auto applicationIdentity = readIdentity(childPath(packagePath, "application"),
                                          "Application manifest root");
  if (!applicationIdentity)
    return applicationIdentity.takeError();
  ArtifactRootReference deploymentReference{
      deployment::deploymentSchema.identity.str(),
      deployment::deploymentSchema.version, *deploymentIdentity};
  ArtifactRootReference manifestReference{
      applicationRuntimeManifestSchema.identity.str(),
      applicationRuntimeManifestSchema.version, *applicationIdentity};
  auto manifest =
      importApplicationRuntimeManifest(manifestReference, artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  auto deployment =
      deployment::importDeployment(deploymentReference, artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  auto closure =
      deriveApplicationPackageClosure(*manifest, *deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  std::set<std::string> artifactNames;
  for (const ArtifactRootReference &root : closure->artifacts)
    artifactNames.insert(formatArtifactIdentityHex(root.artifact));
  std::set<std::string> blobNames;
  for (const BlobDigest &digest : closure->blobs)
    blobNames.insert(formatBlobDigestHex(digest));
  if (llvm::Error error = validateStoreEntries(
          childPath(packagePath, "objects"), artifactNames))
    return std::move(error);
  if (llvm::Error error =
          validateStoreEntries(childPath(packagePath, "blobs"), blobNames))
    return std::move(error);
  return ImportedApplicationPackage(std::move(*manifest),
                                    std::move(*deployment));
}

llvm::Error
publishApplicationPackage(const ApplicationDeploymentArtifacts &application,
                          llvm::StringRef outputPath,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  if (outputPath.empty())
    return invalid("application package output path is empty");
  auto closure = deriveApplicationPackageClosure(
      application.runtimeManifest, application.deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();

  llvm::SmallString<256> destination(outputPath);
  llvm::sys::path::remove_dots(destination, true);
  const llvm::StringRef filename = llvm::sys::path::filename(destination);
  if (filename.empty() || filename == "." || filename == "..")
    return invalid("application package output has no directory name");
  llvm::SmallString<256> parent = llvm::sys::path::parent_path(destination);
  if (parent.empty())
    parent = ".";
  llvm::SmallString<256> model(parent);
  llvm::sys::path::append(model,
                          ("." + filename + ".loom-application-package").str());
  llvm::SmallString<256> staging;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(model, staging))
    return ioError("cannot create application package staging directory: " +
                   error.message());
  bool stagingExists = true;
  llvm::scope_exit cleanup([&] {
    if (stagingExists)
      std::filesystem::remove_all(staging.str().str());
  });

  const std::string objects = childPath(staging, "objects");
  const std::string blobDirectory = childPath(staging, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directory(objects))
    return ioError("cannot create application objects directory: " +
                   error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(blobDirectory))
    return ioError("cannot create application blobs directory: " +
                   error.message());
  if (llvm::Error error =
          writeText(childPath(staging, "root"),
                    formatArtifactIdentityHex(
                        application.deployment.reference().artifact)))
    return error;
  if (llvm::Error error =
          writeText(childPath(staging, "application"),
                    formatArtifactIdentityHex(
                        application.runtimeManifest.reference().artifact)))
    return error;
  for (const ArtifactRootReference &root : closure->artifacts) {
    auto bytes = artifacts.getStoredObject(root);
    if (!bytes)
      return bytes.takeError();
    if (llvm::Error error = writeFile(
            childPath(objects, formatArtifactIdentityHex(root.artifact)),
            *bytes))
      return error;
  }
  for (const BlobDigest &digest : closure->blobs) {
    auto bytes = blobs.get(digest);
    if (!bytes)
      return bytes.takeError();
    if (llvm::Error error = writeFile(
            childPath(blobDirectory, formatBlobDigestHex(digest)), *bytes))
      return error;
  }
  auto imported = importApplicationPackage(staging);
  if (!imported)
    return invalid("staged package failed independent import: " +
                   llvm::toString(imported.takeError()));
  if (llvm::Error error = publishNoReplace(staging, destination))
    return error;
  stagingExists = false;
  return llvm::Error::success();
}

} // namespace loom::application
