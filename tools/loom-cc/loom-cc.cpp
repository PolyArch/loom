//===-- loom-cc.cpp - Loom C/C++ frontend driver --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is derived from clang/tools/driver/driver.cpp in the LLVM project.
//
//===----------------------------------------------------------------------===//
//
// Loom frontend driver. This is a thin wrapper around clang's libDriver that
// behaves as a drop-in replacement for gcc/g++ on the same argument list.
// The implementation mirrors upstream clang/tools/driver/driver.cpp; only
// upstream behavior unrelated to the standard compile pipeline (BOLT,
// CL-mode environment-variable injection unrelated to detection, etc.) has
// been trimmed.
//
// argv[0] basename selects the driver mode: invocations whose name ends in
// "++", "cxx", or "c++" run in g++ mode; everything else runs in gcc mode.
// A loom-c++ symlink to loom-cc is produced at build time so that the same
// binary covers both languages.
//
// Compilation jobs project source candidate annotations to nonsemantic LLVM
// metadata before optimization. Object-producing jobs additionally embed the
// frontend-owned relocatable accelerator payload. Preprocessing, syntax
// checking, and dependency-only jobs retain their ordinary forms.
//
//===----------------------------------------------------------------------===//

#include "Application/ProductBuild.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Stack.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Options/Options.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#if LLVM_ON_UNIX
#include <signal.h>
#endif

using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;

static const char *GetStableCStr(llvm::StringSet<> &Saved, llvm::StringRef S);

namespace {

llvm::Error productError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

struct LoomDriverOptions final {
  std::string accelerationProfile;
  std::string hardwarePath;
  std::string visualizationPath;
  std::string localToolConfigPath;
  std::string deploymentPath;
  std::string mappingTechCandidateLimit;
  std::string mappingWallTimeLimitMilliseconds;
  std::string mappingRepairCandidateLimit;
  std::string mappingStoppingPolicy;
  std::string mappingSpectrumEndpoint;
  std::string portfolioManifestPath;
  std::string portfolioRepositoryRoot;
  std::string portfolioCacheRoot;
  std::string portfolioApplicationIdentity;
  std::string portfolioInputName;
  std::string fpaWeightRootPath;
  std::string fpaArtifactStorePath;
  std::string fpaBlobStorePath;
  std::string fpaConditionsPath;
  std::vector<std::string> operatorProtocolSymbols;

  bool requestsProductFlow() const {
    return !hardwarePath.empty() || !visualizationPath.empty() ||
           !localToolConfigPath.empty() || !deploymentPath.empty() ||
           !mappingTechCandidateLimit.empty() ||
           !mappingWallTimeLimitMilliseconds.empty() ||
           !mappingRepairCandidateLimit.empty() ||
           !mappingStoppingPolicy.empty() || !mappingSpectrumEndpoint.empty() ||
           !portfolioManifestPath.empty() || !portfolioRepositoryRoot.empty() ||
           !portfolioCacheRoot.empty() ||
           !portfolioApplicationIdentity.empty() ||
           !portfolioInputName.empty() || !fpaWeightRootPath.empty() ||
           !fpaArtifactStorePath.empty() || !fpaBlobStorePath.empty() ||
           !fpaConditionsPath.empty() || !operatorProtocolSymbols.empty();
  }
};

llvm::Expected<bool> consumeLoomOption(llvm::StringRef argument,
                                       llvm::StringRef name, std::size_t &index,
                                       llvm::ArrayRef<const char *> arguments,
                                       std::set<std::string> &seen,
                                       std::string &value) {
  llvm::StringRef parsed;
  if (argument == name) {
    if (index + 1 == arguments.size() || arguments[index + 1] == nullptr)
      return productError("loom_driver_option_invalid",
                          name + " requires a value");
    parsed = arguments[++index];
  } else if (argument.consume_front(name) && argument.consume_front("=")) {
    parsed = argument;
  } else {
    return false;
  }
  if (!seen.insert(name.str()).second)
    return productError("loom_driver_option_invalid",
                        name + " may appear only once");
  if (parsed.empty())
    return productError("loom_driver_option_invalid",
                        name + " requires a nonempty value");
  value = parsed.str();
  return true;
}

llvm::Expected<LoomDriverOptions>
extractLoomDriverOptions(llvm::SmallVectorImpl<const char *> &arguments) {
  LoomDriverOptions options;
  std::set<std::string> seen;
  llvm::SmallVector<const char *, 256> retained;
  if (!arguments.empty())
    retained.push_back(arguments.front());
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    if (arguments[index] == nullptr)
      continue;
    const llvm::StringRef argument(arguments[index]);
    auto profile =
        consumeLoomOption(argument, "--loom-accel-profile", index, arguments,
                          seen, options.accelerationProfile);
    if (!profile)
      return profile.takeError();
    if (*profile)
      continue;
    auto hardware = consumeLoomOption(argument, "--loom-hardware", index,
                                      arguments, seen, options.hardwarePath);
    if (!hardware)
      return hardware.takeError();
    if (*hardware)
      continue;
    auto visualization =
        consumeLoomOption(argument, "--loom-viz-export", index, arguments, seen,
                          options.visualizationPath);
    if (!visualization)
      return visualization.takeError();
    if (*visualization)
      continue;
    auto localToolConfig =
        consumeLoomOption(argument, "--loom-local-config", index, arguments,
                          seen, options.localToolConfigPath);
    if (!localToolConfig)
      return localToolConfig.takeError();
    if (*localToolConfig)
      continue;
    auto deployment =
        consumeLoomOption(argument, "--loom-deploy-output", index, arguments,
                          seen, options.deploymentPath);
    if (!deployment)
      return deployment.takeError();
    if (*deployment)
      continue;
    auto techCandidateLimit = consumeLoomOption(
        argument, "--loom-mapping-tech-candidate-limit", index, arguments, seen,
        options.mappingTechCandidateLimit);
    if (!techCandidateLimit)
      return techCandidateLimit.takeError();
    if (*techCandidateLimit)
      continue;
    auto mappingWallTimeLimit = consumeLoomOption(
        argument, "--loom-mapping-wall-time-limit-ms", index, arguments, seen,
        options.mappingWallTimeLimitMilliseconds);
    if (!mappingWallTimeLimit)
      return mappingWallTimeLimit.takeError();
    if (*mappingWallTimeLimit)
      continue;
    auto mappingRepairCandidateLimit = consumeLoomOption(
        argument, "--loom-mapping-repair-candidate-limit", index, arguments,
        seen, options.mappingRepairCandidateLimit);
    if (!mappingRepairCandidateLimit)
      return mappingRepairCandidateLimit.takeError();
    if (*mappingRepairCandidateLimit)
      continue;
    auto mappingStoppingPolicy =
        consumeLoomOption(argument, "--loom-mapping-stopping-policy", index,
                          arguments, seen, options.mappingStoppingPolicy);
    if (!mappingStoppingPolicy)
      return mappingStoppingPolicy.takeError();
    if (*mappingStoppingPolicy)
      continue;
    auto mappingSpectrumEndpoint =
        consumeLoomOption(argument, "--loom-mapping-spectrum-endpoint", index,
                          arguments, seen, options.mappingSpectrumEndpoint);
    if (!mappingSpectrumEndpoint)
      return mappingSpectrumEndpoint.takeError();
    if (*mappingSpectrumEndpoint)
      continue;
    auto portfolioManifest =
        consumeLoomOption(argument, "--loom-portfolio-manifest", index,
                          arguments, seen, options.portfolioManifestPath);
    if (!portfolioManifest)
      return portfolioManifest.takeError();
    if (*portfolioManifest)
      continue;
    auto portfolioRepositoryRoot =
        consumeLoomOption(argument, "--loom-portfolio-repository-root", index,
                          arguments, seen, options.portfolioRepositoryRoot);
    if (!portfolioRepositoryRoot)
      return portfolioRepositoryRoot.takeError();
    if (*portfolioRepositoryRoot)
      continue;
    auto portfolioCacheRoot =
        consumeLoomOption(argument, "--loom-portfolio-cache-root", index,
                          arguments, seen, options.portfolioCacheRoot);
    if (!portfolioCacheRoot)
      return portfolioCacheRoot.takeError();
    if (*portfolioCacheRoot)
      continue;
    auto portfolioApplication = consumeLoomOption(
        argument, "--loom-portfolio-application", index, arguments, seen,
        options.portfolioApplicationIdentity);
    if (!portfolioApplication)
      return portfolioApplication.takeError();
    if (*portfolioApplication)
      continue;
    auto portfolioInput =
        consumeLoomOption(argument, "--loom-portfolio-input", index, arguments,
                          seen, options.portfolioInputName);
    if (!portfolioInput)
      return portfolioInput.takeError();
    if (*portfolioInput)
      continue;
    auto fpaWeight =
        consumeLoomOption(argument, "--loom-fpa-weight-root", index, arguments,
                          seen, options.fpaWeightRootPath);
    if (!fpaWeight)
      return fpaWeight.takeError();
    if (*fpaWeight)
      continue;
    auto fpaArtifacts =
        consumeLoomOption(argument, "--loom-fpa-artifact-store", index,
                          arguments, seen, options.fpaArtifactStorePath);
    if (!fpaArtifacts)
      return fpaArtifacts.takeError();
    if (*fpaArtifacts)
      continue;
    auto fpaBlobs =
        consumeLoomOption(argument, "--loom-fpa-blob-store", index, arguments,
                          seen, options.fpaBlobStorePath);
    if (!fpaBlobs)
      return fpaBlobs.takeError();
    if (*fpaBlobs)
      continue;
    auto fpaConditions =
        consumeLoomOption(argument, "--loom-fpa-conditions", index, arguments,
                          seen, options.fpaConditionsPath);
    if (!fpaConditions)
      return fpaConditions.takeError();
    if (*fpaConditions)
      continue;
    std::string protocolSymbol;
    std::set<std::string> protocolOptionSeen;
    auto protocol =
        consumeLoomOption(argument, "--loom-operator-protocol-symbol", index,
                          arguments, protocolOptionSeen, protocolSymbol);
    if (!protocol)
      return protocol.takeError();
    if (*protocol) {
      if (llvm::is_contained(options.operatorProtocolSymbols, protocolSymbol))
        return productError("loom_driver_option_invalid",
                            "operator protocol symbol is duplicated");
      options.operatorProtocolSymbols.push_back(std::move(protocolSymbol));
      continue;
    }
    retained.push_back(arguments[index]);
  }
  arguments.assign(retained.begin(), retained.end());
  if (!options.hardwarePath.empty() && !options.accelerationProfile.empty())
    return productError("loom_driver_option_invalid",
                        "external hardware and an acceleration profile are "
                        "mutually exclusive");
  if (options.requestsProductFlow() && options.deploymentPath.empty())
    return productError("loom_driver_option_unsupported",
                        "product options require a Deployment output");
  return options;
}

bool preventsFinalLink(const llvm::opt::ArgList &arguments) {
  return arguments.hasArg(options::OPT_E) || arguments.hasArg(options::OPT_S) ||
         arguments.hasArg(options::OPT_c) ||
         arguments.hasArg(options::OPT_emit_llvm) ||
         arguments.hasArg(options::OPT_fsyntax_only) ||
         arguments.hasArg(options::OPT_M, options::OPT_MM) ||
         arguments.hasArg(options::OPT__HASH_HASH_HASH);
}

llvm::Expected<std::uint64_t>
parsePositiveProductLimit(llvm::StringRef spelling, llvm::StringRef option) {
  std::uint64_t value = 0;
  if (spelling.getAsInteger(10, value) || value == 0)
    return productError("loom_driver_option_invalid",
                        option + " requires a positive integer");
  return value;
}

llvm::Expected<loom::application::ProductBuildOptions>
makeProductBuildOptions(const LoomDriverOptions &options) {
  std::uint64_t techCandidateLimit =
      loom::application::defaultProductTechCandidateLimit;
  if (!options.mappingTechCandidateLimit.empty()) {
    auto parsed =
        parsePositiveProductLimit(options.mappingTechCandidateLimit,
                                  "--loom-mapping-tech-candidate-limit");
    if (!parsed)
      return parsed.takeError();
    techCandidateLimit = *parsed;
  }
  std::uint64_t wallTimeLimit =
      loom::application::defaultProductMappingWallTimeLimitMilliseconds;
  if (!options.mappingWallTimeLimitMilliseconds.empty()) {
    auto parsed =
        parsePositiveProductLimit(options.mappingWallTimeLimitMilliseconds,
                                  "--loom-mapping-wall-time-limit-ms");
    if (!parsed)
      return parsed.takeError();
    wallTimeLimit = *parsed;
  }
  std::optional<std::uint64_t> mappingRepairCandidateLimit;
  if (!options.mappingRepairCandidateLimit.empty()) {
    auto parsed =
        parsePositiveProductLimit(options.mappingRepairCandidateLimit,
                                  "--loom-mapping-repair-candidate-limit");
    if (!parsed)
      return parsed.takeError();
    mappingRepairCandidateLimit = *parsed;
  }
  auto stoppingPolicy = loom::application::parseProductMappingStoppingPolicy(
      options.mappingStoppingPolicy.empty() ? "first_verified"
                                            : options.mappingStoppingPolicy);
  if (!stoppingPolicy)
    return stoppingPolicy.takeError();
  auto spectrumEndpoint =
      loom::application::parseProductMappingSpectrumEndpoint(
          options.mappingSpectrumEndpoint.empty()
              ? "automatic"
              : options.mappingSpectrumEndpoint);
  if (!spectrumEndpoint)
    return spectrumEndpoint.takeError();
  return loom::application::ProductBuildOptions{
      options.deploymentPath,
      options.accelerationProfile,
      options.hardwarePath,
      options.visualizationPath,
      options.localToolConfigPath,
      options.operatorProtocolSymbols,
      techCandidateLimit,
      wallTimeLimit,
      mappingRepairCandidateLimit,
      *stoppingPolicy,
      *spectrumEndpoint,
      options.portfolioManifestPath,
      options.portfolioRepositoryRoot,
      options.portfolioCacheRoot,
      options.portfolioApplicationIdentity,
      options.portfolioInputName,
      options.fpaWeightRootPath,
      options.fpaArtifactStorePath,
      options.fpaBlobStorePath,
      options.fpaConditionsPath};
}

llvm::StringRef projectedValue(llvm::ArrayRef<std::string> projection,
                               llvm::StringRef prefix) {
  for (const std::string &argument : projection) {
    llvm::StringRef value(argument);
    if (value.consume_front(prefix))
      return value;
  }
  return {};
}

llvm::Error
validateUserTargetArguments(llvm::ArrayRef<const char *> arguments,
                            llvm::ArrayRef<std::string> projection) {
  const llvm::StringRef targetTriple = projectedValue(projection, "--target=");
  const llvm::StringRef architecture = projectedValue(projection, "-march=");
  const llvm::StringRef abi = projectedValue(projection, "-mabi=");
  const llvm::StringRef codeModel = projectedValue(projection, "-mcmodel=");
  const llvm::StringRef backendCpu = projectedValue(projection, "-mcpu=");
  if (targetTriple.empty() || architecture.empty() || abi.empty() ||
      codeModel.empty() || backendCpu.empty())
    return productError("loom_product_driver_projection_invalid",
                        "driver argument projection omits a target field");
  auto requireEqual = [&](llvm::StringRef kind, llvm::StringRef selected,
                          llvm::StringRef required) -> llvm::Error {
    if (selected == required)
      return llvm::Error::success();
    return productError("loom_product_target_conflict",
                        kind + " selects '" + selected +
                            "' but the System requires '" + required + "'");
  };
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    if (!arguments[index])
      continue;
    llvm::StringRef argument(arguments[index]);
    if (argument == "-target" || argument == "--target") {
      if (++index == arguments.size() || !arguments[index])
        return productError("loom_product_target_conflict",
                            "target option has no value");
      if (llvm::Triple::normalize(arguments[index]) != targetTriple)
        return productError("loom_product_target_conflict",
                            "target triple disagrees with the System");
      continue;
    }
    if (argument.starts_with("--target=") || argument.starts_with("-target=")) {
      const llvm::StringRef value = argument.drop_front(argument.find('=') + 1);
      if (llvm::Triple::normalize(value) != targetTriple)
        return productError("loom_product_target_conflict",
                            "target triple disagrees with the System");
      continue;
    }
    if (argument.consume_front("-march=")) {
      if (llvm::Error error =
              requireEqual("architecture", argument, architecture))
        return error;
      continue;
    }
    if (argument.consume_front("-mabi=")) {
      if (llvm::Error error = requireEqual("ABI", argument, abi))
        return error;
      continue;
    }
    if (argument.consume_front("-mcmodel=")) {
      if (llvm::Error error = requireEqual("code model", argument, codeModel))
        return error;
      continue;
    }
    if (argument.consume_front("-mcpu=")) {
      if (llvm::Error error = requireEqual("backend CPU", argument, backendCpu))
        return error;
      continue;
    }
    if (argument == "-fno-lto" || argument == "-fno-fat-lto-objects" ||
        argument.starts_with("-Wl,--plugin-opt=-mattr=") ||
        argument.starts_with("-Wl,--unresolved-symbols="))
      return productError("loom_product_target_conflict",
                          "option conflicts with exact final-link import");
    if (argument.starts_with("-flto=") && argument != "-flto=full")
      return productError("loom_product_target_conflict",
                          "Deployment requires full LTO");
    if (argument.starts_with("-fuse-ld=") && argument != "-fuse-ld=lld")
      return productError("loom_product_target_conflict",
                          "Deployment requires the pinned LLD provider");
  }
  return llvm::Error::success();
}

void insertProductTargetArguments(
    llvm::SmallVectorImpl<const char *> &arguments, llvm::StringSet<> &saved,
    llvm::ArrayRef<std::string> projection) {
  for (const std::string &argument : projection)
    arguments.push_back(GetStableCStr(saved, argument));
}

} // namespace

std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    llvm::SmallString<128> ExecutablePath(Argv0);
    if (!llvm::sys::fs::exists(ExecutablePath))
      if (llvm::ErrorOr<std::string> P =
              llvm::sys::findProgramByName(ExecutablePath))
        ExecutablePath = *P;
    return std::string(ExecutablePath);
  }
  void *P = (void *)(intptr_t)GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

static const char *GetStableCStr(llvm::StringSet<> &Saved, llvm::StringRef S) {
  return Saved.insert(S).first->getKeyData();
}

namespace {

llvm::Expected<std::string>
findProductLinkOutput(const Compilation &compilation) {
  std::set<std::string> outputs;
  for (const Command &command : compilation.getJobs()) {
    if (command.getSource().getKind() != Action::LinkJobClass)
      continue;
    for (const std::string &output : command.getOutputFilenames())
      if (!output.empty())
        outputs.insert(output);
  }
  if (outputs.size() != 1)
    return productError("loom_final_link_invalid",
                        "Deployment requires exactly one final link output");
  return *outputs.begin();
}

struct ProductLinkStaging final {
  std::string publicOutput;
  std::string stagedOutput;
};

llvm::Expected<ProductLinkStaging>
prepareProductLinkStaging(const Compilation &compilation,
                          llvm::SmallVectorImpl<const char *> &args,
                          llvm::StringSet<> &saved) {
  auto output = findProductLinkOutput(compilation);
  if (!output)
    return output.takeError();
  const llvm::StringRef filename = llvm::sys::path::filename(*output);
  if (filename.empty())
    return productError("loom_final_link_invalid",
                        "Deployment final link output has no filename");

  llvm::SmallString<256> stagingDirectory(*output);
  stagingDirectory += ".loom-link";
  if (std::error_code error =
          llvm::sys::fs::remove_directories(stagingDirectory))
    return productError("loom_final_link_staging_failed",
                        "cannot replace final-link staging directory: " +
                            error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(stagingDirectory))
    return productError("loom_final_link_staging_failed",
                        "cannot create final-link staging directory: " +
                            error.message());

  llvm::SmallString<256> stagedOutput(stagingDirectory);
  llvm::sys::path::append(stagedOutput, filename);
  args.push_back(GetStableCStr(saved, "-save-temps=obj"));
  args.push_back(GetStableCStr(saved, "-o"));
  args.push_back(GetStableCStr(saved, stagedOutput));
  return ProductLinkStaging{std::move(*output), stagedOutput.str().str()};
}

llvm::Error publishProductLink(const ProductLinkStaging &staging) {
  const auto require = [&](llvm::StringRef suffix) -> llvm::Error {
    if (llvm::sys::fs::exists(staging.stagedOutput + suffix.str()))
      return llvm::Error::success();
    return productError("loom_final_link_artifact_missing",
                        "staged final-link artifact is missing: " +
                            staging.stagedOutput + suffix);
  };
  for (llvm::StringRef suffix :
       {llvm::StringRef(), llvm::StringRef(".resolution.txt"),
        llvm::StringRef(".0.5.precodegen.bc")})
    if (llvm::Error error = require(suffix))
      return error;

  const auto publish = [&](llvm::StringRef suffix) -> llvm::Error {
    const std::string source = staging.stagedOutput + suffix.str();
    const std::string destination = staging.publicOutput + suffix.str();
    if (std::error_code error = llvm::sys::fs::rename(source, destination))
      return productError("loom_final_link_publication_failed",
                          "cannot publish final-link artifact '" + source +
                              "' as '" + destination + "': " + error.message());
    return llvm::Error::success();
  };
  for (llvm::StringRef suffix :
       {llvm::StringRef(".0.5.precodegen.bc"), llvm::StringRef(),
        llvm::StringRef(".resolution.txt")})
    if (llvm::Error error = publish(suffix))
      return error;
  return llvm::Error::success();
}

} // namespace

extern int cc1_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(llvm::ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr,
                                  const llvm::ToolContext &);

static void insertTargetAndModeArgs(const ParsedClangName &NameParts,
                                    llvm::SmallVectorImpl<const char *> &ArgV,
                                    llvm::StringSet<> &Saved) {
  int Insert = 0;
  if (!ArgV.empty())
    ++Insert;
  if (NameParts.DriverMode) {
    ArgV.insert(ArgV.begin() + Insert,
                GetStableCStr(Saved, NameParts.DriverMode));
  }
  if (NameParts.TargetIsValid) {
    const char *Pair[] = {"-target",
                          GetStableCStr(Saved, NameParts.TargetPrefix)};
    ArgV.insert(ArgV.begin() + Insert, std::begin(Pair), std::end(Pair));
  }
}

enum class CC1ProgramAction : std::uint8_t {
  Other,
  Assembly,
  LLVM,
  Object,
};

struct CC1CommandSemantics final {
  CC1ProgramAction action = CC1ProgramAction::Other;
};

bool isCC1Command(const Command &command) {
  return !command.getArguments().empty() && command.getArguments().front() &&
         llvm::StringRef(command.getArguments().front()) == "-cc1";
}

std::optional<CC1CommandSemantics> cc1CommandSemantics(const Command &command) {
  if (!isCC1Command(command))
    return CC1CommandSemantics{};
  DiagnosticOptions options;
  IgnoringDiagConsumer consumer;
  DiagnosticsEngine diagnostics(DiagnosticIDs::create(), options, &consumer,
                                /*ShouldOwnClient=*/false);
  CompilerInvocation invocation;
  if (!CompilerInvocation::CreateFromArgs(invocation, command.getArguments(),
                                          diagnostics))
    return std::nullopt;
  CC1CommandSemantics semantics;
  switch (invocation.getFrontendOpts().ProgramAction) {
  case frontend::EmitAssembly:
    semantics.action = CC1ProgramAction::Assembly;
    break;
  case frontend::EmitBC:
  case frontend::EmitLLVM:
    semantics.action = CC1ProgramAction::LLVM;
    break;
  case frontend::EmitObj:
    semantics.action = CC1ProgramAction::Object;
    break;
  default:
    break;
  }
  return semantics;
}

bool feedsAssemblerAction(const Compilation &compilation,
                          const Command &producer) {
  for (const std::string &output : producer.getOutputFilenames()) {
    if (output.empty())
      continue;
    for (const Command &consumer : compilation.getJobs()) {
      if (&consumer == &producer ||
          consumer.getSource().getKind() != Action::AssembleJobClass ||
          isCC1Command(consumer))
        continue;
      if (llvm::any_of(consumer.getInputInfos(), [&](const InputInfo &input) {
            return input.isFilename() && input.getFilename() == output;
          }))
        return true;
    }
  }
  return false;
}

llvm::Error appendFrontendPassPlugins(Compilation &compilation,
                                      llvm::StringSet<> &saved) {
  const std::string payloadOption =
      (llvm::Twine("-fpass-plugin=") + LOOM_RELOCATABLE_PAYLOAD_PASS_PATH)
          .str();
  const std::string candidateOption =
      (llvm::Twine("-fpass-plugin=") + LOOM_CANDIDATE_PROJECTION_PASS_PATH)
          .str();
  for (Command &command : compilation.getJobs()) {
    auto semantics = cc1CommandSemantics(command);
    if (!semantics || semantics->action == CC1ProgramAction::Other)
      continue;
    llvm::opt::ArgStringList arguments(command.getArguments().begin(),
                                       command.getArguments().end());
    if (semantics->action == CC1ProgramAction::Object ||
        (semantics->action == CC1ProgramAction::Assembly &&
         feedsAssemblerAction(compilation, command)) ||
        (semantics->action == CC1ProgramAction::LLVM &&
         command.getSource().getType() == types::TY_LTO_BC))
      arguments.push_back(GetStableCStr(saved, payloadOption));
    arguments.push_back(GetStableCStr(saved, candidateOption));
    command.replaceArguments(std::move(arguments));
  }
  return llvm::Error::success();
}

template <class T>
static T checkEnvVar(const char *EnvOptSet, const char *EnvOptFile,
                     std::string &OptFile) {
  const char *Str = ::getenv(EnvOptSet);
  if (!Str)
    return T{};
  T OptVal = Str;
  if (const char *Var = ::getenv(EnvOptFile))
    OptFile = Var;
  return OptVal;
}

static void SetBackdoorDriverOutputsFromEnvVars(Driver &D) {
  D.CCPrintOptions = checkEnvVar<bool>(
      "CC_PRINT_OPTIONS", "CC_PRINT_OPTIONS_FILE", D.CCPrintOptionsFilename);
  D.CCLogDiagnostics =
      checkEnvVar<bool>("CC_LOG_DIAGNOSTICS", "CC_LOG_DIAGNOSTICS_FILE",
                        D.CCLogDiagnosticsFilename);
  D.CCPrintProcessStats =
      checkEnvVar<bool>("CC_PRINT_PROC_STAT", "CC_PRINT_PROC_STAT_FILE",
                        D.CCPrintStatReportFilename);
  D.CCPrintInternalStats =
      checkEnvVar<bool>("CC_PRINT_INTERNAL_STAT", "CC_PRINT_INTERNAL_STAT_FILE",
                        D.CCPrintInternalStatReportFilename);
}

constexpr llvm::StringLiteral candidatePragmaReplayArgument =
    "-loom-internal-candidate-replay=";

bool containsReservedCandidateReplayArgument(
    llvm::ArrayRef<const char *> arguments) {
  unsigned missingArgumentIndex = 0;
  unsigned missingArgumentCount = 0;
  llvm::opt::InputArgList parsed = clang::getDriverOptTable().ParseArgs(
      arguments, missingArgumentIndex, missingArgumentCount,
      llvm::opt::Visibility(clang::options::CC1Option));
  return llvm::any_of(parsed.filtered(clang::options::OPT_UNKNOWN),
                      [&](const llvm::opt::Arg *argument) {
                        return llvm::StringRef(argument->getAsString(parsed))
                            .starts_with(candidatePragmaReplayArgument);
                      });
}

llvm::Error
rejectReservedCandidateReplayArguments(const Compilation &compilation) {
  for (const Command &command : compilation.getJobs())
    if (containsReservedCandidateReplayArgument(command.getArguments()))
      return productError("loom_internal_invocation_invalid",
                          "candidate replay argument is unsupported");
  return llvm::Error::success();
}

llvm::Error normalizeCC1ResponseArguments(Compilation &compilation,
                                          llvm::BumpPtrAllocator &allocator,
                                          llvm::vfs::FileSystem &fileSystem) {
  for (Command &command : compilation.getJobs()) {
    llvm::opt::ArgStringList arguments(command.getArguments().begin(),
                                       command.getArguments().end());
    if (arguments.empty() || llvm::StringRef(arguments.front()) != "-cc1")
      continue;
    if (llvm::none_of(arguments, [](const char *argument) {
          return argument && llvm::StringRef(argument).starts_with("@") &&
                 llvm::StringRef(argument).size() > 1;
        }))
      continue;
    llvm::cl::ExpansionContext context(
        allocator, llvm::cl::TokenizeGNUCommandLine, &fileSystem);
    if (llvm::Error error = context.expandResponseFiles(arguments))
      return productError("loom_internal_invocation_invalid",
                          "cc1 response arguments cannot be normalized: " +
                              llvm::toString(std::move(error)));
    command.replaceArguments(std::move(arguments));
  }
  return llvm::Error::success();
}

static int ExecuteCC1Tool(llvm::SmallVectorImpl<const char *> &ArgV,
                          const llvm::ToolContext &Ctx,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  // Reset cl::opt counts because options are global state and the driver may
  // already have parsed them in this process.
  llvm::cl::ResetAllOptionOccurrences();

  llvm::BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, llvm::cl::TokenizeGNUCommandLine,
                                  VFS.get());
  if (llvm::Error Err = ECtx.expandResponseFiles(ArgV)) {
    llvm::errs() << toString(std::move(Err)) << '\n';
    return 1;
  }
  llvm::StringRef Tool = ArgV[1];
  void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
  if (Tool == "-cc1") {
    if (containsReservedCandidateReplayArgument(ArgV)) {
      llvm::errs() << "loom-cc: error: loom_internal_invocation_invalid: "
                      "candidate replay argument is unsupported\n";
      return 1;
    }
    auto enableSandbox = llvm::sys::sandbox::scopedEnable();
    return cc1_main(llvm::ArrayRef(ArgV).slice(1), ArgV[0],
                    GetExecutablePathVP);
  }
  if (Tool == "-cc1as") {
    auto enableSandbox = llvm::sys::sandbox::scopedEnable();
    return cc1as_main(llvm::ArrayRef(ArgV).slice(2), ArgV[0],
                      GetExecutablePathVP);
  }
  if (Tool == "-cc1gen-reproducer") {
    auto enableSandbox = llvm::sys::sandbox::scopedEnable();
    return cc1gen_reproducer_main(llvm::ArrayRef(ArgV).slice(2), ArgV[0],
                                  GetExecutablePathVP, Ctx);
  }
  llvm::errs() << "error: unknown integrated tool '" << Tool << "'. "
               << "Valid tools include '-cc1', '-cc1as' and "
                  "'-cc1gen-reproducer'.\n";
  return 1;
}

static int loom_main(int Argc, char **Argv,
                     const llvm::ToolContext &ToolContext) {
  noteBottomOfStack();
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace, preprocessed source, and "
      "associated run script.\n");

  llvm::SmallVector<const char *, 256> Args(Argv, Argv + Argc);
  if (llvm::sys::Process::FixupStandardFileDescriptors())
    return 1;
  llvm::InitializeAllTargets();

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);

  const char *ProgName =
      ToolContext.NeedsPrependArg ? ToolContext.PrependArg : ToolContext.Path;

  bool ClangCLMode =
      IsClangCL(getDriverMode(ProgName, llvm::ArrayRef(Args).slice(1)));

  auto VFS = llvm::vfs::getRealFileSystem();
  if (llvm::Error Err = expandResponseFiles(Args, ClangCLMode, A, VFS.get())) {
    llvm::errs() << toString(std::move(Err)) << '\n';
    return 1;
  }

  // -cc1 family: fast path through the integrated tool dispatcher.
  if (Args.size() >= 2 && llvm::StringRef(Args[1]).starts_with("-cc1"))
    return ExecuteCC1Tool(Args, ToolContext, VFS);

  auto LoomOptions = extractLoomDriverOptions(Args);
  if (!LoomOptions) {
    llvm::errs() << "loom-cc: error: "
                 << llvm::toString(LoomOptions.takeError()) << '\n';
    return 1;
  }
  bool CanonicalPrefixes = true;
  for (int i = 1, size = Args.size(); i < size; ++i) {
    if (Args[i] == nullptr)
      continue;
    if (llvm::StringRef(Args[i]) == "-canonical-prefixes")
      CanonicalPrefixes = true;
    else if (llvm::StringRef(Args[i]) == "-no-canonical-prefixes")
      CanonicalPrefixes = false;
  }

  llvm::StringSet<> SavedStrings;
  if (const char *Override = ::getenv("CCC_OVERRIDE_OPTIONS")) {
    driver::applyOverrideOptions(Args, Override, SavedStrings,
                                 "CCC_OVERRIDE_OPTIONS", &llvm::errs());
  }
  if (!LoomOptions->portfolioManifestPath.empty() && Args.size() != 1) {
    llvm::errs() << "loom-cc: error: a portfolio selection derives its exact "
                    "sources and compiler options from the manifest\n";
    return 1;
  }

  std::vector<std::string> ProductTargetArguments;
  std::unique_ptr<loom::application::ProductBuildInvocation> ProductInvocation;
  if (LoomOptions->requestsProductFlow()) {
    auto ProductOptions = makeProductBuildOptions(*LoomOptions);
    if (!ProductOptions) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(ProductOptions.takeError()) << '\n';
      return 1;
    }
    auto Invocation = loom::application::ProductBuildInvocation::create(
        std::move(*ProductOptions));
    if (!Invocation) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(Invocation.takeError()) << '\n';
      return 1;
    }
    auto Projection = (*Invocation)->compilerArguments();
    if (llvm::Error Error = validateUserTargetArguments(Args, Projection)) {
      llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(Error))
                   << '\n';
      return 1;
    }
    ProductTargetArguments = std::move(Projection);
    ProductInvocation = std::move(*Invocation);
  }

  std::string Path = GetExecutablePath(ToolContext.Path, CanonicalPrefixes);

  // CLANG_SPAWN_CC1 controls whether cc1 runs in-process or as a fresh
  // subprocess. We honor the upstream default plus -f{,no-}integrated-cc1.
  bool UseNewCC1Process = CLANG_SPAWN_CC1;
  for (const char *Arg : Args)
    UseNewCC1Process = llvm::StringSwitch<bool>(Arg)
                           .Case("-fno-integrated-cc1", true)
                           .Case("-fintegrated-cc1", false)
                           .Default(UseNewCC1Process);

  std::unique_ptr<DiagnosticOptions> DiagOpts = CreateAndPopulateDiagOpts(Args);
  DiagOpts->DiagnosticSuppressionMappingsFile.clear();

  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), *DiagOpts);
  llvm::StringRef ExeBasename(llvm::sys::path::stem(ProgName));
  DiagClient->setPrefix(std::string(ExeBasename));

  DiagnosticsEngine Diags(DiagnosticIDs::create(), *DiagOpts, DiagClient);

  if (!DiagOpts->DiagnosticSerializationFile.empty()) {
    auto SerializedConsumer = clang::serialized_diags::create(
        DiagOpts->DiagnosticSerializationFile, *DiagOpts,
        /*MergeChildRecords=*/true);
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.takeClient(), std::move(SerializedConsumer)));
  }
  ProcessWarningOptions(Diags, *DiagOpts, *VFS, /*ReportDiags=*/false);

  Driver TheDriver(Path, llvm::sys::getDefaultTargetTriple(), Diags,
                   /*Title=*/"loom-cc LLVM compiler", VFS);
  auto TargetAndMode = ToolChain::getTargetAndModeFromProgramName(ProgName);
  TheDriver.setTargetAndMode(TargetAndMode);
  if (ToolContext.NeedsPrependArg || CanonicalPrefixes)
    TheDriver.setPrependArg(ToolContext.PrependArg);

  insertTargetAndModeArgs(TargetAndMode, Args, SavedStrings);
  if (!ProductTargetArguments.empty())
    insertProductTargetArguments(Args, SavedStrings, ProductTargetArguments);
  SetBackdoorDriverOutputsFromEnvVars(TheDriver);

  auto ExecuteCC1WithContext =
      [&ToolContext, &VFS](llvm::SmallVectorImpl<const char *> &ArgV) {
        return ExecuteCC1Tool(ArgV, ToolContext, VFS);
      };
  if (!UseNewCC1Process) {
    TheDriver.CC1Main = ExecuteCC1WithContext;
    llvm::CrashRecoveryContext::Enable(
        /*NeedsPOSIXUtilitySignalHandling=*/true);
  }

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  std::optional<ProductLinkStaging> ProductLink;
  if (LoomOptions->requestsProductFlow() && C && !C->containsError()) {
    auto Staging = prepareProductLinkStaging(*C, Args, SavedStrings);
    if (!Staging) {
      llvm::errs() << "loom-cc: error: " << llvm::toString(Staging.takeError())
                   << '\n';
      return 1;
    }
    ProductLink = std::move(*Staging);
    C.reset(TheDriver.BuildCompilation(Args));
    if (C && !C->containsError()) {
      auto Output = findProductLinkOutput(*C);
      if (!Output) {
        llvm::errs() << "loom-cc: error: " << llvm::toString(Output.takeError())
                     << '\n';
        return 1;
      }
      if (*Output != ProductLink->stagedOutput) {
        llvm::errs() << "loom-cc: error: staged final link output changed\n";
        return 1;
      }
    }
  }
  const bool printsCommands =
      C && C->getArgs().hasArg(options::OPT__HASH_HASH_HASH);
  if (C && !C->containsError()) {
    if (!printsCommands) {
      if (llvm::Error error = normalizeCC1ResponseArguments(*C, A, *VFS)) {
        llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(error))
                     << '\n';
        return 1;
      }
      if (llvm::Error error = rejectReservedCandidateReplayArguments(*C)) {
        llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(error))
                     << '\n';
        return 1;
      }
    }
    if (!LoomOptions->deploymentPath.empty() &&
        preventsFinalLink(C->getArgs())) {
      llvm::errs() << "loom-cc: error: Deployment output requires a final "
                      "link invocation\n";
      return 1;
    }
    if (!printsCommands)
      if (llvm::Error error = appendFrontendPassPlugins(*C, SavedStrings)) {
        llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(error))
                     << '\n';
        return 1;
      }
  }

  Driver::ReproLevel ReproLevel = Driver::ReproLevel::OnCrash;
  if (Arg *A = C->getArgs().getLastArg(options::OPT_gen_reproducer_eq)) {
    auto Level =
        llvm::StringSwitch<std::optional<Driver::ReproLevel>>(A->getValue())
            .Case("off", Driver::ReproLevel::Off)
            .Case("crash", Driver::ReproLevel::OnCrash)
            .Case("error", Driver::ReproLevel::OnError)
            .Case("always", Driver::ReproLevel::Always)
            .Default(std::nullopt);
    if (!Level) {
      llvm::errs() << "Unknown value for " << A->getSpelling() << ": '"
                   << A->getValue() << "'\n";
      return 1;
    }
    ReproLevel = *Level;
  }

  int Res = 1;
  bool IsCrash = false;
  Driver::CommandStatus CommandStatus = Driver::CommandStatus::Ok;
  const Command *FailingCommand = nullptr;
  int CommandRes = 0;
  if (!C->getJobs().empty())
    FailingCommand = &*C->getJobs().begin();
  if (C && !C->containsError()) {
    llvm::SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
    Res = TheDriver.ExecuteCompilation(*C, FailingCommands);
    for (const auto &P : FailingCommands) {
      CommandRes = P.first;
      FailingCommand = P.second;
      if (!Res)
        Res = CommandRes;
      IsCrash = CommandRes < 0 || CommandRes == 70;
#if LLVM_ON_UNIX
      IsCrash |= CommandRes > 128;
#endif
      CommandStatus =
          IsCrash ? Driver::CommandStatus::Crash : Driver::CommandStatus::Error;
      if (IsCrash)
        break;
    }
  }

  if (Res == 0 && ProductLink) {
    if (llvm::Error Error = publishProductLink(*ProductLink)) {
      llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(Error))
                   << '\n';
      Res = 1;
    } else if (llvm::Error Error = ProductInvocation->buildFromFinalLink(
                   ProductLink->publicOutput)) {
      llvm::errs() << "loom-cc: error: " << llvm::toString(std::move(Error))
                   << '\n';
      Res = 1;
    }
  }

  if (FailingCommand != nullptr &&
      TheDriver.maybeGenerateCompilationDiagnostics(CommandStatus, ReproLevel,
                                                    *C, *FailingCommand))
    Res = 1;

  if (!UseNewCC1Process && IsCrash) {
    llvm::BuryPointer(llvm::TimerGroup::acquireTimerGlobals());
  } else {
    llvm::TimerGroup::printAll(llvm::errs());
    llvm::TimerGroup::clearAll();
  }

#if LLVM_ON_UNIX
  if (CommandRes > 128 && CommandRes != 255) {
    llvm::sys::unregisterHandlers();
    Diags.getClient()->~DiagnosticConsumer();
    raise(CommandRes - 128);
  }
  if (CommandRes == -2) {
    llvm::sys::unregisterHandlers();
    Diags.getClient()->~DiagnosticConsumer();
    raise(SIGABRT);
  }
#endif

  return Res;
}

int main(int Argc, char **Argv) {
  llvm::ToolContext Ctx{Argv[0], /*PrependArg=*/nullptr,
                        /*NeedsPrependArg=*/false};
  return loom_main(Argc, Argv, Ctx);
}
