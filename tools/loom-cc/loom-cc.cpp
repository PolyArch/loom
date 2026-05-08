//===-- loom-cc.cpp - Loom C/C++ frontend driver --------------------------===//
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
// Loom-specific frontend behavior (pragma/attribute parsing, IR-metadata
// emission) is intentionally not present at this point. The driver is wired
// in here so that subsequent work can attach a metadata pass at the cc1
// boundary or via a clang plugin without touching the build graph again.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Stack.h"
#include "clang/Config/config.h"
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
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <memory>
#include <optional>
#if LLVM_ON_UNIX
#include <signal.h>
#endif

using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;

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

extern int cc1_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(llvm::ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr,
                                  const llvm::ToolContext &);

// TODO(loom-frontend): hook for Loom metadata pragmas / attributes
// A future change wires Loom-specific pragma/attribute parsing and the
// IR-metadata pass into the cc1 boundary (either by registering a plugin
// before invoking cc1_main, or by intercepting BuildCompilation jobs to
// thread an extra cc1 phase). The driver entry point is left identical to
// upstream so that hook can be added without disturbing the gcc/g++ ABI.

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
