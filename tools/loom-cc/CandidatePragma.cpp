#include "Frontend/Raising/CandidateHints.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::raising::CandidateHintErrorKind;

enum class PendingCandidateTarget : std::uint8_t {
  FunctionOrInvalid,
  Loop,
};

struct PendingCandidatePragma final {
  clang::SourceLocation location;
  PendingCandidateTarget target = PendingCandidateTarget::FunctionOrInvalid;
  clang::SourceLocation loopBegin;
  std::uint64_t marker = 0;
};

struct CandidatePragmaState final {
  std::vector<PendingCandidatePragma> pending;
  std::uint64_t nextMarker = 1;
};

thread_local llvm::DenseMap<clang::Preprocessor *, CandidatePragmaState>
    pendingCandidatePragmas;
thread_local llvm::DenseMap<clang::Preprocessor *, bool>
    candidateProjectionConsumers;

CandidatePragmaState &candidatePragmaState(clang::Preprocessor &preprocessor) {
  return pendingCandidatePragmas[&preprocessor];
}

void clearPendingCandidatePragma(clang::Preprocessor &preprocessor) {
  pendingCandidatePragmas.erase(&preprocessor);
  candidateProjectionConsumers.erase(&preprocessor);
}

bool isCandidateProjectionAction(clang::frontend::ActionKind action) {
  switch (action) {
  case clang::frontend::EmitAssembly:
  case clang::frontend::EmitBC:
  case clang::frontend::EmitLLVM:
  case clang::frontend::EmitObj:
    return true;
  default:
    return false;
  }
}

void reportRaw(clang::DiagnosticsEngine &diagnostics,
               clang::SourceLocation location, llvm::StringRef message) {
  const unsigned id = diagnostics.getDiagnosticIDs()->getCustomDiagID(
      clang::DiagnosticIDs::Error, message);
  diagnostics.Report(location, id);
}

void report(clang::DiagnosticsEngine &diagnostics,
            clang::SourceLocation location, CandidateHintErrorKind kind,
            llvm::StringRef message) {
  reportRaw(diagnostics, location,
            (llvm::Twine("candidate_hint_") +
             loom::raising::candidateHintErrorKindName(kind) + ": " + message)
                .str());
}

void consumeDirective(clang::Preprocessor &preprocessor, clang::Token &token) {
  while (token.isNot(clang::tok::eod))
    preprocessor.LexUnexpandedToken(token);
}

void reinject(clang::Preprocessor &preprocessor, clang::Token token,
              bool disableMacroExpansion) {
  auto tokens = std::make_unique<clang::Token[]>(1);
  tokens[0] = token;
  preprocessor.EnterTokenStream(std::move(tokens), 1, disableMacroExpansion,
                                /*IsReinject=*/disableMacroExpansion);
}

void injectLoopMarker(clang::Preprocessor &preprocessor,
                      clang::SourceLocation pragma,
                      llvm::StringRef encodedMarker, clang::Token loopToken) {
  constexpr unsigned markerTokenCount = 11;
  auto tokens = std::make_unique<clang::Token[]>(markerTokenCount + 1);
  for (unsigned index = 0; index != markerTokenCount; ++index) {
    tokens[index].startToken();
    tokens[index].setLocation(pragma);
  }
  tokens[0].setKind(clang::tok::kw_switch);
  tokens[1].setKind(clang::tok::l_paren);
  tokens[2].setKind(clang::tok::identifier);
  tokens[2].setIdentifierInfo(
      preprocessor.getIdentifierInfo("__builtin_annotation"));
  tokens[3].setKind(clang::tok::l_paren);
  tokens[4].setKind(clang::tok::numeric_constant);
  preprocessor.CreateString("0", tokens[4], pragma, pragma);
  tokens[5].setKind(clang::tok::comma);
  tokens[6].setKind(clang::tok::string_literal);
  std::string literal = (llvm::Twine("\"") + encodedMarker + "\"").str();
  preprocessor.CreateString(literal, tokens[6], pragma, pragma);
  tokens[7].setKind(clang::tok::r_paren);
  tokens[8].setKind(clang::tok::r_paren);
  tokens[9].setKind(clang::tok::kw_default);
  tokens[10].setKind(clang::tok::colon);
  tokens[markerTokenCount] = loopToken;
  preprocessor.EnterTokenStream(std::move(tokens), markerTokenCount + 1,
                                /*DisableMacroExpansion=*/true,
                                /*IsReinject=*/true);
}

bool isLoopToken(const clang::Token &token) {
  return token.isOneOf(clang::tok::kw_for, clang::tok::kw_while,
                       clang::tok::kw_do);
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
             CandidateHintErrorKind::InvalidEncoding,
             "expected 'candidate' after '#pragma loom'");
      consumeDirective(preprocessor, token);
      return;
    }
    if (introducer.Kind != clang::PIK_HashPragma ||
        introducer.Loc.isMacroID()) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' cannot use a macro pragma introducer");
      preprocessor.LexUnexpandedToken(token);
      consumeDirective(preprocessor, token);
      return;
    }

    preprocessor.LexUnexpandedToken(token);
    if (token.isNot(clang::tok::eod)) {
      report(preprocessor.getDiagnostics(), token.getLocation(),
             CandidateHintErrorKind::InvalidEncoding,
             "unexpected tokens after '#pragma loom candidate'");
      consumeDirective(preprocessor, token);
      return;
    }

    clang::Token next;
    preprocessor.LexUnexpandedToken(next);
    if (next.is(clang::tok::identifier) && next.getIdentifierInfo() &&
        preprocessor.getMacroInfo(next.getIdentifierInfo())) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support a macro-expanded "
             "target boundary");
      reinject(preprocessor, next, /*disableMacroExpansion=*/false);
      return;
    }
    const bool targetsLoop = isLoopToken(next);
    auto consumer = candidateProjectionConsumers.find(&preprocessor);
    const bool hasConsumer = consumer != candidateProjectionConsumers.end();
    if (!hasConsumer) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             CandidateHintErrorKind::UnsupportedConstruct,
             "preprocessing-only actions cannot preserve exact candidate "
             "source ranges");
      reinject(preprocessor, next, /*disableMacroExpansion=*/true);
      return;
    }
    if (!targetsLoop) {
      if (hasConsumer) {
        candidatePragmaState(preprocessor)
            .pending.push_back(PendingCandidatePragma{
                introducer.Loc,
                PendingCandidateTarget::FunctionOrInvalid,
                {},
                0});
        reinject(preprocessor, next, /*disableMacroExpansion=*/false);
      } else {
        reinject(preprocessor, next, /*disableMacroExpansion=*/false);
      }
      return;
    }
    if (next.getLocation().isMacroID()) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support a macro-expanded "
             "loop boundary");
      reinject(preprocessor, next, /*disableMacroExpansion=*/false);
      return;
    }
    if (!hasConsumer) {
      reinject(preprocessor, next, /*disableMacroExpansion=*/true);
      return;
    }
    CandidatePragmaState &state = candidatePragmaState(preprocessor);
    if (state.nextMarker == 0) {
      report(preprocessor.getDiagnostics(), introducer.Loc,
             CandidateHintErrorKind::ProjectionProofNotEstablished,
             "'#pragma loom candidate' exhausted its loop marker domain");
      reinject(preprocessor, next, /*disableMacroExpansion=*/true);
      return;
    }
    const std::uint64_t marker = state.nextMarker++;
    state.pending.push_back({introducer.Loc, PendingCandidateTarget::Loop,
                             next.getLocation(), marker});
    if (!consumer->second) {
      reinject(preprocessor, next, /*disableMacroExpansion=*/true);
      return;
    }
    injectLoopMarker(preprocessor, introducer.Loc,
                     loom::raising::encodeLoopCandidateMarker(marker), next);
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
      : compiler(compiler), projectCandidates(isCandidateProjectionAction(
                                compiler.getFrontendOpts().ProgramAction)) {
    clearPendingCandidatePragma(compiler.getPreprocessor());
    candidateProjectionConsumers.try_emplace(&compiler.getPreprocessor(),
                                             projectCandidates);
  }

  ~CandidatePragmaConsumer() override {
    clearPendingCandidatePragma(compiler.getPreprocessor());
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef declarations) override {
    CandidatePragmaState &state =
        candidatePragmaState(compiler.getPreprocessor());
    if (state.pending.empty())
      return true;

    llvm::SmallVector<clang::Decl *> explicitDeclarations;
    for (clang::Decl *declaration : declarations)
      if (!declaration->isImplicit() && declaration->getBeginLoc().isValid())
        explicitDeclarations.push_back(declaration);
    if (explicitDeclarations.empty())
      return true;

    clang::SourceManager &sourceManager = compiler.getSourceManager();
    clang::DiagnosticsEngine &diagnostics = compiler.getDiagnostics();

    clang::Decl *first = explicitDeclarations.front();
    clang::Decl *last = explicitDeclarations.back();
    std::vector<PendingCandidatePragma> pending;
    std::vector<PendingCandidatePragma> deferred;
    for (PendingCandidatePragma &candidate : state.pending) {
      if (sourceManager.isBeforeInTranslationUnit(last->getEndLoc(),
                                                  candidate.location))
        deferred.push_back(std::move(candidate));
      else
        pending.push_back(std::move(candidate));
    }
    state.pending = std::move(deferred);

    llvm::SmallVector<const PendingCandidatePragma *> functionCandidates;
    llvm::DenseMap<unsigned, unsigned> loopCandidateCounts;
    auto precedesDeclaration = [&](const PendingCandidatePragma &candidate) {
      return !sourceManager.isBeforeInTranslationUnit(first->getBeginLoc(),
                                                      candidate.location);
    };
    for (const PendingCandidatePragma &candidate : pending) {
      if (precedesDeclaration(candidate)) {
        functionCandidates.push_back(&candidate);
        continue;
      }
      if (candidate.target == PendingCandidateTarget::Loop)
        ++loopCandidateCounts[candidate.loopBegin.getRawEncoding()];
    }

    if (functionCandidates.size() == 1) {
      handleFunctionCandidate(*functionCandidates.front(), explicitDeclarations,
                              diagnostics, sourceManager);
    } else {
      for (const PendingCandidatePragma *candidate : functionCandidates)
        report(diagnostics, candidate->location,
               CandidateHintErrorKind::InvalidPlacement,
               "multiple '#pragma loom candidate' directives select one "
               "function definition");
    }

    for (const PendingCandidatePragma &candidate : pending) {
      if (precedesDeclaration(candidate))
        continue;
      if (candidate.target == PendingCandidateTarget::Loop &&
          loopCandidateCounts.lookup(candidate.loopBegin.getRawEncoding()) >
              1) {
        report(diagnostics, candidate.location,
               CandidateHintErrorKind::InvalidPlacement,
               "multiple '#pragma loom candidate' directives select one loop");
        continue;
      }
      handleLoopCandidate(candidate, explicitDeclarations, diagnostics,
                          sourceManager);
    }
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &) override {
    CandidatePragmaState &state =
        candidatePragmaState(compiler.getPreprocessor());
    if (!state.pending.empty()) {
      for (const PendingCandidatePragma &pending : state.pending)
        report(compiler.getDiagnostics(), pending.location,
               CandidateHintErrorKind::InvalidPlacement,
               "dangling '#pragma loom candidate' has no function or loop");
      state.pending.clear();
    }
  }

private:
  static const clang::Stmt *findLoop(const clang::FunctionDecl &function,
                                     clang::SourceLocation begin,
                                     const clang::SourceManager &sourceManager,
                                     bool &nestedCallable) {
    const clang::Stmt *body = function.getBody();
    if (!body)
      return nullptr;
    const clang::Stmt *found = nullptr;
    auto visit = [&](const auto &self, const clang::Stmt *statement,
                     bool nested) -> void {
      if (found || !statement)
        return;
      if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt,
                    clang::CXXForRangeStmt>(statement) &&
          sourceManager.getExpansionLoc(statement->getBeginLoc()) ==
              sourceManager.getExpansionLoc(begin)) {
        found = statement;
        nestedCallable = nested;
        return;
      }
      nested |= llvm::isa<clang::LambdaExpr>(statement);
      for (const clang::Stmt *child : statement->children())
        self(self, child, nested);
    };
    visit(visit, body, false);
    return found;
  }

  static bool containsExternallyOwnedSwitchLabel(const clang::Stmt &loop) {
    bool found = false;
    auto visit = [&](const auto &self, const clang::Stmt *statement) -> void {
      if (found || !statement)
        return;
      if (statement != &loop && llvm::isa<clang::SwitchStmt>(statement))
        return;
      if (llvm::isa<clang::SwitchCase>(statement)) {
        found = true;
        return;
      }
      for (const clang::Stmt *child : statement->children())
        self(self, child);
    };
    visit(visit, &loop);
    return found;
  }

  clang::SourceLocation exclusiveStatementEnd(const clang::Stmt &statement) {
    clang::SourceLocation end = statement.getEndLoc();
    clang::SourceManager &sourceManager = compiler.getSourceManager();
    const clang::LangOptions &language = compiler.getLangOpts();
    clang::Token endToken;
    if (!clang::Lexer::getRawToken(end, endToken, sourceManager, language,
                                   /*IgnoreWhiteSpace=*/true) &&
        endToken.isNot(clang::tok::semi) &&
        endToken.isNot(clang::tok::r_brace)) {
      clang::SourceLocation afterSemicolon =
          clang::Lexer::findLocationAfterToken(
              end, clang::tok::semi, sourceManager, language,
              /*SkipTrailingWhitespaceAndNewLine=*/false);
      if (afterSemicolon.isValid())
        return afterSemicolon;
    }
    return clang::Lexer::getLocForEndOfToken(end, 0, sourceManager, language);
  }

  void handleFunctionCandidate(const PendingCandidatePragma &candidate,
                               llvm::ArrayRef<clang::Decl *> declarations,
                               clang::DiagnosticsEngine &diagnostics,
                               clang::SourceManager &sourceManager) {
    const clang::SourceLocation pragmaLocation = candidate.location;
    if (candidate.target == PendingCandidateTarget::Loop) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::InvalidPlacement,
             "'#pragma loom candidate' loop marker is outside a function");
      return;
    }
    if (declarations.size() != 1) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::InvalidPlacement,
             "'#pragma loom candidate' must select exactly one function "
             "definition");
      return;
    }
    if (llvm::isa<clang::FunctionTemplateDecl>(declarations.front())) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support function templates");
      return;
    }
    auto *function = llvm::dyn_cast<clang::FunctionDecl>(declarations.front());
    if (!function || !function->doesThisDeclarationHaveABody()) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::InvalidPlacement,
             "'#pragma loom candidate' must immediately precede a function "
             "definition");
      return;
    }
    if (llvm::isa<clang::CXXMethodDecl>(function)) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support member functions");
      return;
    }
    clang::SourceLocation targetBegin = function->getBeginLoc();
    clang::SourceLocation targetEnd = clang::Lexer::getLocForEndOfToken(
        function->getEndLoc(), 0, sourceManager, compiler.getLangOpts());
    auto range = sourceRange(pragmaLocation, targetBegin, targetEnd, "function",
                             diagnostics, sourceManager);
    if (!range)
      return;
    auto carrier = sourcePosition(sourceManager, function->getLocation());
    if (!carrier) {
      llvm::consumeError(carrier.takeError());
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::ProjectionProofNotEstablished,
             "'#pragma loom candidate' function carrier is unavailable");
      return;
    }
    loom::raising::FunctionCandidateAnnotation hint{
        range->sourceFile, *carrier, range->pragma, range->begin, range->end};
    accept(*function, pragmaLocation,
           loom::raising::encodeFunctionCandidateAnnotation(hint), diagnostics);
  }

  void handleLoopCandidate(const PendingCandidatePragma &candidate,
                           llvm::ArrayRef<clang::Decl *> declarations,
                           clang::DiagnosticsEngine &diagnostics,
                           clang::SourceManager &sourceManager) {
    if (candidate.target != PendingCandidateTarget::Loop) {
      if (declarations.size() == 1 &&
          llvm::isa<clang::FunctionDecl>(declarations.front()))
        report(diagnostics, candidate.location,
               CandidateHintErrorKind::InvalidPlacement,
               "'#pragma loom candidate' must immediately precede a function "
               "definition or loop");
      else
        report(diagnostics, candidate.location,
               CandidateHintErrorKind::UnsupportedConstruct,
               "'#pragma loom candidate' function is nested in an unsupported "
               "declaration context");
      return;
    }
    if (declarations.size() != 1) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::InvalidPlacement,
             "'#pragma loom candidate' loop must belong to one function "
             "definition");
      return;
    }
    auto *function = llvm::dyn_cast<clang::FunctionDecl>(declarations.front());
    if (!function) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' loop is nested in an unsupported "
             "declaration context");
      return;
    }
    if (llvm::isa<clang::CXXMethodDecl>(function)) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support member-function loop "
             "carriers");
      return;
    }
    if (!function->doesThisDeclarationHaveABody()) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::InvalidPlacement,
             "'#pragma loom candidate' must immediately precede a function "
             "definition or loop");
      return;
    }
    bool nestedCallable = false;
    const clang::Stmt *loop =
        findLoop(*function, candidate.loopBegin, sourceManager, nestedCallable);
    if (!loop) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::ProjectionProofNotEstablished,
             "'#pragma loom candidate' loop identity is unavailable");
      return;
    }
    if (nestedCallable) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support lambda-local loops");
      return;
    }
    if (containsExternallyOwnedSwitchLabel(*loop)) {
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::UnsupportedConstruct,
             "'#pragma loom candidate' does not support a loop body with a "
             "switch label owned outside the loop");
      return;
    }
    clang::SourceLocation targetEnd = exclusiveStatementEnd(*loop);
    auto range = sourceRange(candidate.location, loop->getBeginLoc(), targetEnd,
                             "loop", diagnostics, sourceManager);
    if (!range)
      return;
    auto carrier = sourcePosition(sourceManager, function->getLocation());
    if (!carrier) {
      llvm::consumeError(carrier.takeError());
      report(diagnostics, candidate.location,
             CandidateHintErrorKind::ProjectionProofNotEstablished,
             "'#pragma loom candidate' function carrier is unavailable");
      return;
    }
    loom::raising::LoopCandidateAnnotation hint{
        candidate.marker, range->sourceFile, *carrier,
        range->pragma,    range->begin,      range->end};
    accept(*function, candidate.location,
           loom::raising::encodeLoopCandidateAnnotation(hint), diagnostics);
  }

  struct CandidateSourceRange final {
    std::string sourceFile;
    loom::raising::SourcePosition pragma;
    loom::raising::SourcePosition begin;
    loom::raising::SourcePosition end;
  };

  std::optional<CandidateSourceRange>
  sourceRange(clang::SourceLocation pragmaLocation,
              clang::SourceLocation targetBegin,
              clang::SourceLocation targetEnd, llvm::StringRef targetName,
              clang::DiagnosticsEngine &diagnostics,
              clang::SourceManager &sourceManager) {
    if (targetBegin.isMacroID() || targetEnd.isMacroID() ||
        targetEnd.isInvalid()) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::UnsupportedConstruct,
             (llvm::Twine("'#pragma loom candidate' does not support a "
                          "macro-expanded ") +
              targetName + " boundary")
                 .str());
      return std::nullopt;
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
             CandidateHintErrorKind::InvalidPlacement,
             (llvm::Twine("'#pragma loom candidate' and its ") + targetName +
              " must share one source file")
                 .str());
      return std::nullopt;
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
             CandidateHintErrorKind::ProjectionProofNotEstablished,
             "'#pragma loom candidate' source range is unavailable");
      return std::nullopt;
    }
    return CandidateSourceRange{pragmaPresumed.getFilename(), *pragma, *begin,
                                *end};
  }

  void accept(clang::FunctionDecl &function,
              clang::SourceLocation pragmaLocation,
              llvm::Expected<std::string> encoded,
              clang::DiagnosticsEngine &diagnostics) {
    if (!encoded) {
      const std::string message = llvm::toString(encoded.takeError());
      reportRaw(diagnostics, pragmaLocation, message);
      return;
    }
    if (projectCandidates && compiler.getCodeGenOpts().DisableLLVMPasses) {
      report(diagnostics, pragmaLocation,
             CandidateHintErrorKind::UnsupportedConstruct,
             "candidate projection requires the LLVM pass pipeline");
      return;
    }
    if (!projectCandidates)
      return;
    attach(function, std::move(*encoded));
  }

  void attach(clang::FunctionDecl &function, llvm::StringRef encoded) {
    const bool sourceRequired =
        function.getASTContext().DeclMustBeEmitted(&function) ||
        (compiler.getLangOpts().EmitAllDecls &&
         function.hasAttr<clang::AnnotateAttr>());
    function.addAttr(clang::AnnotateAttr::CreateImplicit(
        function.getASTContext(), encoded, nullptr, 0));
    const bool mayBeDeferred =
        !function.isExternallyVisible() || function.isInlined();
    const bool emittedAtO0 = compiler.getLangOpts().EmitAllDecls &&
                             !compiler.getCodeGenOpts().isOptimizedBuild();
    if (mayBeDeferred && !emittedAtO0 && needsTemporaryRetention() &&
        !function.hasAttr<clang::RetainAttr>() &&
        !function.hasAttr<clang::UsedAttr>()) {
      function.addAttr(clang::AnnotateAttr::CreateImplicit(
          function.getASTContext(),
          sourceRequired
              ? loom::raising::
                    candidateSourceRequiredTemporaryRetentionAnnotation
              : loom::raising::candidateTemporaryRetentionAnnotation,
          nullptr, 0));
      function.addAttr(
          clang::UsedAttr::CreateImplicit(function.getASTContext()));
    }
  }

  bool needsTemporaryRetention() const {
    switch (compiler.getFrontendOpts().ProgramAction) {
    case clang::frontend::EmitBC:
    case clang::frontend::EmitLLVM:
    case clang::frontend::EmitObj:
      return true;
    case clang::frontend::EmitAssembly:
      return llvm::is_contained(compiler.getCodeGenOpts().PassPlugins,
                                LOOM_RELOCATABLE_PAYLOAD_PASS_PATH);
    default:
      return false;
    }
  }

  clang::CompilerInstance &compiler;
  bool projectCandidates = false;
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
