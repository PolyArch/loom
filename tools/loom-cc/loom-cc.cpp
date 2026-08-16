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
// Object-producing frontend jobs embed the frontend-owned relocatable
// accelerator payload through Loom's LLVM module pass. Preprocessing, syntax
// checking, LLVM IR output, and assembly output retain their ordinary forms.
//
//===----------------------------------------------------------------------===//

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
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
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

static const char *GetStableCStr(llvm::StringSet<> &Saved,
                                 llvm::StringRef S);

namespace {

llvm::Error productError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

struct LoomDriverOptions final {
  std::string accelerationProfile;
  std::string hardwarePath;
  std::string visualizationPath;
  std::string deploymentPath;

  bool requestsProductFlow() const {
    return !hardwarePath.empty() || !visualizationPath.empty() ||
           !deploymentPath.empty();
  }
};

llvm::Expected<bool> consumeLoomOption(
    llvm::StringRef argument, llvm::StringRef name, std::size_t &index,
    llvm::ArrayRef<const char *> arguments, std::set<std::string> &seen,
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
    auto profile = consumeLoomOption(
        argument, "--loom-accel-profile", index, arguments, seen,
        options.accelerationProfile);
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
    auto visualization = consumeLoomOption(
        argument, "--loom-viz-export", index, arguments, seen,
        options.visualizationPath);
    if (!visualization)
      return visualization.takeError();
    if (*visualization)
      continue;
    auto deployment = consumeLoomOption(
        argument, "--loom-deploy-output", index, arguments, seen,
        options.deploymentPath);
    if (!deployment)
      return deployment.takeError();
    if (*deployment)
      continue;
    retained.push_back(arguments[index]);
  }
  arguments.assign(retained.begin(), retained.end());
  if (!options.hardwarePath.empty() &&
      !options.accelerationProfile.empty())
    return productError("loom_driver_option_invalid",
                        "external hardware and an acceleration profile are "
                        "mutually exclusive");
  if (options.requestsProductFlow() && options.deploymentPath.empty())
    return productError("loom_driver_option_unsupported",
                        "hardware and visualization bindings currently require "
                        "a Deployment output");
  return options;
}

bool preventsFinalLink(llvm::ArrayRef<const char *> arguments) {
  for (const char *raw : arguments.drop_front()) {
    if (!raw)
      continue;
    const llvm::StringRef argument(raw);
    if (argument == "-E" || argument == "-S" || argument == "-c" ||
        argument == "-emit-llvm" || argument == "-fsyntax-only" ||
        argument == "-M" || argument == "-MM" || argument == "-###")
      return true;
  }
  return false;
}

llvm::Expected<std::vector<std::string>> readProductDriverArguments(
    llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return productError("loom_product_driver_projection_invalid",
                        "cannot read driver argument projection: " +
                            buffer.getError().message());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (bytes.empty() || bytes.back() != '\0')
    return productError("loom_product_driver_projection_invalid",
                        "driver argument projection is not terminated");
  std::vector<std::string> result;
  while (!bytes.empty()) {
    const std::size_t end = bytes.find('\0');
    if (end == 0 || end == llvm::StringRef::npos)
      return productError("loom_product_driver_projection_invalid",
                          "driver argument projection contains an empty "
                          "argument");
    result.push_back(bytes.take_front(end).str());
    bytes = bytes.drop_front(end + 1);
  }
  return result;
}

llvm::SmallVector<std::string, 8>
productHelperOptions(const LoomDriverOptions &options) {
  llvm::SmallVector<std::string, 8> result;
  result.push_back("--deployment-output=" + options.deploymentPath);
  if (!options.accelerationProfile.empty())
    result.push_back("--acceleration-profile=" +
                     options.accelerationProfile);
  if (!options.hardwarePath.empty())
    result.push_back("--hardware=" + options.hardwarePath);
  if (!options.visualizationPath.empty())
    result.push_back("--visualization=" + options.visualizationPath);
  return result;
}

llvm::Error invokeProductHelper(const LoomDriverOptions &options,
                                llvm::StringRef action) {
  llvm::SmallVector<std::string, 8> owned = productHelperOptions(options);
  owned.push_back(action.str());
  llvm::SmallVector<llvm::StringRef, 10> command{LOOM_APPLICATION_BUILD_PATH};
  for (const std::string &argument : owned)
    command.push_back(argument);
  std::string message;
  bool failed = false;
  const int status = llvm::sys::ExecuteAndWait(
      LOOM_APPLICATION_BUILD_PATH, command, std::nullopt, {}, 0, 0, &message,
      &failed);
  if (failed || status < 0)
    return productError("loom_product_helper_unavailable",
                        "cannot execute application build helper: " + message);
  if (status != 0)
    return productError("loom_product_build_failed",
                        "application build helper exited with status " +
                            llvm::Twine(status));
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::string>>
requestProductDriverArguments(const LoomDriverOptions &options) {
  int descriptor = -1;
  llvm::SmallString<256> path;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "loom-product-driver", "args", descriptor, path))
    return productError("loom_product_driver_projection_invalid",
                        "cannot create driver argument capture: " +
                            error.message());
  llvm::sys::Process::SafelyCloseFileDescriptor(descriptor);
  llvm::FileRemover remove(path);
  const std::string action =
      (llvm::Twine("--driver-arguments-output=") + path).str();
  if (llvm::Error error = invokeProductHelper(options, action))
    return std::move(error);
  return readProductDriverArguments(path);
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

llvm::Error validateUserTargetArguments(
    llvm::ArrayRef<const char *> arguments,
    llvm::ArrayRef<std::string> projection) {
  const llvm::StringRef targetTriple =
      projectedValue(projection, "--target=");
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
    if (argument.starts_with("--target=") ||
        argument.starts_with("-target=")) {
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
      if (llvm::Error error =
              requireEqual("code model", argument, codeModel))
        return error;
      continue;
    }
    if (argument.consume_front("-mcpu=")) {
      if (llvm::Error error =
              requireEqual("backend CPU", argument, backendCpu))
        return error;
      continue;
    }
    if (argument == "-fno-lto" || argument == "-fno-fat-lto-objects" ||
        argument.starts_with("-Wl,--plugin-opt=-mattr="))
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

static bool shouldEmbedRelocatablePayload(llvm::ArrayRef<const char *> args) {
  for (const char *rawArg : llvm::ArrayRef(args).drop_front()) {
    if (rawArg == nullptr)
      continue;
    const llvm::StringRef arg(rawArg);
    if (arg == "-E" || arg == "-S" || arg == "-emit-llvm" ||
        arg == "-fsyntax-only" || arg == "-M" || arg == "-MM" || arg == "-###")
      return false;
  }
  return true;
}

static void
insertRelocatablePayloadPass(llvm::SmallVectorImpl<const char *> &args,
                             llvm::StringSet<> &saved) {
  if (!shouldEmbedRelocatablePayload(args))
    return;
  const std::string option =
      std::string("-fpass-plugin=") + LOOM_RELOCATABLE_PAYLOAD_PASS_PATH;
  args.insert(args.begin() + 1, GetStableCStr(saved, option));
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
  D.CCLogDiagnostics = checkEnvVar<bool>(
      "CC_LOG_DIAGNOSTICS", "CC_LOG_DIAGNOSTICS_FILE",
      D.CCLogDiagnosticsFilename);
  D.CCPrintProcessStats = checkEnvVar<bool>(
      "CC_PRINT_PROC_STAT", "CC_PRINT_PROC_STAT_FILE",
      D.CCPrintStatReportFilename);
  D.CCPrintInternalStats = checkEnvVar<bool>(
      "CC_PRINT_INTERNAL_STAT", "CC_PRINT_INTERNAL_STAT_FILE",
      D.CCPrintInternalStatReportFilename);
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
  if (Tool == "-cc1")
    return cc1_main(llvm::ArrayRef(ArgV).slice(1), ArgV[0],
                    GetExecutablePathVP);
  if (Tool == "-cc1as")
    return cc1as_main(llvm::ArrayRef(ArgV).slice(2), ArgV[0],
                      GetExecutablePathVP);
  if (Tool == "-cc1gen-reproducer")
    return cc1gen_reproducer_main(llvm::ArrayRef(ArgV).slice(2), ArgV[0],
                                  GetExecutablePathVP, Ctx);
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
  if (llvm::Error Err =
          expandResponseFiles(Args, ClangCLMode, A, VFS.get())) {
    llvm::errs() << toString(std::move(Err)) << '\n';
    return 1;
  }

  // -cc1 family: fast path through the integrated tool dispatcher.
  if (Args.size() >= 2 && llvm::StringRef(Args[1]).starts_with("-cc1")) {
    auto EnableSandbox = llvm::sys::sandbox::scopedEnable();
    return ExecuteCC1Tool(Args, ToolContext, VFS);
  }

  auto LoomOptions = extractLoomDriverOptions(Args);
  if (!LoomOptions) {
    llvm::errs() << "loom-cc: error: "
                 << llvm::toString(LoomOptions.takeError()) << '\n';
    return 1;
  }
  if (!LoomOptions->deploymentPath.empty() && preventsFinalLink(Args)) {
    llvm::errs() << "loom-cc: error: Deployment output requires a final "
                    "link invocation\n";
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

  std::vector<std::string> ProductTargetArguments;
  if (LoomOptions->requestsProductFlow()) {
    auto Projection = requestProductDriverArguments(*LoomOptions);
    if (!Projection) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(Projection.takeError()) << '\n';
      return 1;
    }
    if (llvm::Error Error =
            validateUserTargetArguments(Args, *Projection)) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(std::move(Error)) << '\n';
      return 1;
    }
    ProductTargetArguments = std::move(*Projection);
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

  std::unique_ptr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(Args);
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
    insertProductTargetArguments(Args, SavedStrings,
                                 ProductTargetArguments);
  insertRelocatablePayloadPass(Args, SavedStrings);
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

  std::optional<std::string> ProductLinkOutput;
  if (LoomOptions->requestsProductFlow() && C && !C->containsError()) {
    auto Output = findProductLinkOutput(*C);
    if (!Output) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(Output.takeError()) << '\n';
      return 1;
    }
    ProductLinkOutput = std::move(*Output);
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
      CommandStatus = IsCrash ? Driver::CommandStatus::Crash
                              : Driver::CommandStatus::Error;
      if (IsCrash)
        break;
    }
  }

  if (Res == 0 && ProductLinkOutput) {
    const std::string action =
        "--final-link-output=" + *ProductLinkOutput;
    if (llvm::Error Error = invokeProductHelper(*LoomOptions, action)) {
      llvm::errs() << "loom-cc: error: "
                   << llvm::toString(std::move(Error)) << '\n';
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
