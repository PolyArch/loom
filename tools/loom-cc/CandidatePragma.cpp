#include "Frontend/Raising/CandidateHints.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct PendingCandidatePragma final {
  clang::SourceLocation location;
};

thread_local llvm::DenseMap<clang::Preprocessor *,
                            std::optional<PendingCandidatePragma>>
    pendingCandidatePragmas;

std::optional<PendingCandidatePragma> &
pendingCandidatePragma(clang::Preprocessor &preprocessor) {
  return pendingCandidatePragmas[&preprocessor];
}

void clearPendingCandidatePragma(clang::Preprocessor &preprocessor) {
  pendingCandidatePragmas.erase(&preprocessor);
}

void report(clang::DiagnosticsEngine &diagnostics,
            clang::SourceLocation location, llvm::StringRef message) {
  const unsigned id = diagnostics.getDiagnosticIDs()->getCustomDiagID(
      clang::DiagnosticIDs::Error, message);
  diagnostics.Report(location, id);
}

void consumeDirective(clang::Preprocessor &preprocessor, clang::Token &token) {
  while (token.isNot(clang::tok::eod))
    preprocessor.LexUnexpandedToken(token);
}

class CandidatePragmaHandler final : public clang::PragmaHandler {
public:
  CandidatePragmaHandler() : PragmaHandler("loom") {}

  void HandlePragma(clang::Preprocessor &preprocessor,
                    clang::PragmaIntroducer introducer,
                    clang::Token &) override {
    clang::Token token;
    preprocessor.LexUnexpandedToken(token);
    if (!token.is(clang::tok::identifier) ||
        token.getIdentifierInfo()->getName() != "candidate") {
      report(preprocessor.getDiagnostics(), token.getLocation(),
             "expected 'candidate' after '#pragma loom'");
      consumeDirective(preprocessor, token);
      return;
    }
    preprocessor.LexUnexpandedToken(token);
    if (token.isNot(clang::tok::eod)) {
      report(preprocessor.getDiagnostics(), token.getLocation(),
             "unexpected tokens after '#pragma loom candidate'");
      consumeDirective(preprocessor, token);
      return;
    }
    if (introducer.Loc.isMacroID()) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             "'#pragma loom candidate' cannot be macro-expanded");
      return;
    }
    auto &pending = pendingCandidatePragma(preprocessor);
    if (pending) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             "another '#pragma loom candidate' is already pending");
      return;
    }
    pending = PendingCandidatePragma{introducer.Loc};
  }
};

llvm::Expected<loom::raising::SourcePosition>
sourcePosition(const clang::SourceManager &sourceManager,
               clang::SourceLocation location) {
  const clang::PresumedLoc presumed = sourceManager.getPresumedLoc(location);
  if (presumed.isInvalid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "source position is unavailable");
  return loom::raising::SourcePosition{presumed.getLine(),
                                       presumed.getColumn()};
}

class CandidatePragmaConsumer final : public clang::ASTConsumer {
public:
  explicit CandidatePragmaConsumer(clang::CompilerInstance &compiler)
      : compiler(compiler) {
    clearPendingCandidatePragma(compiler.getPreprocessor());
  }

  ~CandidatePragmaConsumer() override {
    clearPendingCandidatePragma(compiler.getPreprocessor());
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef declarations) override {
    auto &pending = pendingCandidatePragma(compiler.getPreprocessor());
    if (!pending)
      return true;

    llvm::SmallVector<clang::Decl *> explicitDeclarations;
    for (clang::Decl *declaration : declarations)
      if (!declaration->isImplicit() && declaration->getBeginLoc().isValid())
        explicitDeclarations.push_back(declaration);
    if (explicitDeclarations.empty())
      return true;

    const clang::SourceLocation pragmaLocation = pending->location;
    pending.reset();
    clang::SourceManager &sourceManager = compiler.getSourceManager();
    clang::DiagnosticsEngine &diagnostics = compiler.getDiagnostics();

    clang::Decl *first = explicitDeclarations.front();
    if (!sourceManager.isBeforeInTranslationUnit(pragmaLocation,
                                                 first->getBeginLoc())) {
      if (auto *function = llvm::dyn_cast<clang::FunctionDecl>(first);
          function && function->doesThisDeclarationHaveABody() &&
          hasFollowingLoop(*function, pragmaLocation, sourceManager)) {
        report(diagnostics, pragmaLocation,
               "loop candidate hints are unsupported by this provider");
        return true;
      }
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' is only valid at file scope before a "
             "function definition");
      return true;
    }
    if (explicitDeclarations.size() != 1) {
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' must select exactly one function "
             "definition");
      return true;
    }
    auto *function = llvm::dyn_cast<clang::FunctionDecl>(first);
    if (!function || !function->doesThisDeclarationHaveABody()) {
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' must immediately precede a function "
             "definition");
      return true;
    }
    if (!function->isExternallyVisible() || function->isInlined()) {
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' requires a non-inline externally "
             "visible function definition");
      return true;
    }

    // Clang's LLVM annotation emitter records FunctionDecl::getLocation() in
    // the annotation's line field. Use the same anchor so multiline
    // declarations retain a coherent source-to-LLVM correspondence.
    clang::SourceLocation targetBegin = function->getLocation();
    clang::SourceLocation targetEnd = clang::Lexer::getLocForEndOfToken(
        function->getEndLoc(), 0, sourceManager, compiler.getLangOpts());
    if (targetBegin.isMacroID() || targetEnd.isMacroID() ||
        targetEnd.isInvalid()) {
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' does not support a macro-expanded "
             "function boundary");
      return true;
    }

    const clang::PresumedLoc pragmaPresumed =
        sourceManager.getPresumedLoc(pragmaLocation);
    const clang::PresumedLoc beginPresumed =
        sourceManager.getPresumedLoc(targetBegin);
    const clang::PresumedLoc endPresumed =
        sourceManager.getPresumedLoc(targetEnd);
    if (pragmaPresumed.isInvalid() || beginPresumed.isInvalid() ||
        endPresumed.isInvalid() ||
        llvm::StringRef(pragmaPresumed.getFilename()) !=
            beginPresumed.getFilename() ||
        llvm::StringRef(beginPresumed.getFilename()) !=
            endPresumed.getFilename()) {
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' and its function must share one source "
             "file");
      return true;
    }

    auto pragma = sourcePosition(sourceManager, pragmaLocation);
    auto begin = sourcePosition(sourceManager, targetBegin);
    auto end = sourcePosition(sourceManager, targetEnd);
    if (!pragma || !begin || !end) {
      if (!pragma)
        llvm::consumeError(pragma.takeError());
      if (!begin)
        llvm::consumeError(begin.takeError());
      if (!end)
        llvm::consumeError(end.takeError());
      report(diagnostics, pragmaLocation,
             "'#pragma loom candidate' source range is unavailable");
      return true;
    }

    loom::raising::FunctionCandidateAnnotation hint{
        pragmaPresumed.getFilename(), *pragma, *begin, *end};
    auto encoded = loom::raising::encodeFunctionCandidateAnnotation(hint);
    if (!encoded) {
      const std::string message = llvm::toString(encoded.takeError());
      report(diagnostics, pragmaLocation, message);
      return true;
    }
    function->addAttr(clang::AnnotateAttr::CreateImplicit(
        function->getASTContext(), *encoded, nullptr, 0));
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &) override {
    auto &pending = pendingCandidatePragma(compiler.getPreprocessor());
    if (!pending)
      return;
    report(compiler.getDiagnostics(), pending->location,
           "dangling '#pragma loom candidate' has no function definition");
    pending.reset();
  }

private:
  static bool hasFollowingLoop(const clang::FunctionDecl &function,
                               clang::SourceLocation pragma,
                               const clang::SourceManager &sourceManager) {
    const clang::Stmt *body = function.getBody();
    if (!body)
      return false;
    bool found = false;
    auto visit = [&](const auto &self, const clang::Stmt *statement) -> void {
      if (found || !statement)
        return;
      if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt>(
              statement) &&
          sourceManager.isBeforeInTranslationUnit(pragma,
                                                  statement->getBeginLoc())) {
        found = true;
        return;
      }
      for (const clang::Stmt *child : statement->children())
        self(self, child);
    };
    visit(visit, body);
    return found;
  }

  clang::CompilerInstance &compiler;
};

class CandidatePragmaAction final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef) override {
    return std::make_unique<CandidatePragmaConsumer>(compiler);
  }

  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  void EndSourceFileAction() override {
    clearPendingCandidatePragma(getCompilerInstance().getPreprocessor());
    clang::PluginASTAction::EndSourceFileAction();
  }

  ActionType getActionType() override { return AddBeforeMainAction; }
};

} // namespace

static clang::FrontendPluginRegistry::Add<CandidatePragmaAction>
    candidatePragmaAction("loom-candidate", "project Loom candidate hints");

static clang::PragmaHandlerRegistry::Add<CandidatePragmaHandler>
    candidatePragmaHandler("loom", "parse Loom pragmas");
