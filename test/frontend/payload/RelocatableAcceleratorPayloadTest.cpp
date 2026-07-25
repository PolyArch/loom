#include "Frontend/Payload/RelocatableAcceleratorPayload.h"
#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Common/ResolvedConfig.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "RelocatablePayloadRootCodec.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

/// Returns the typed rejection message, failing the test when the hostile input
/// was accepted instead.
template <typename T>
std::string rejectionMessage(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "a value that must fail closed was accepted");
  return llvm::toString(value.takeError());
}

void requireRejection(const char *test, const std::string &message,
                      llvm::StringRef marker, const std::string &what) {
  if (!llvm::StringRef(message).contains(marker))
    fail(test, what + ": " + message);
}

constexpr llvm::StringRef canonicalTargetTriple = "x86_64-unknown-linux-gnu";
constexpr llvm::StringRef canonicalDataLayout =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-"
    "n8:16:32:64-S128";

/// One representative module carrying module flags, ABI attributes, debug
/// information, source provenance, symbols, linkage, visibility, COMDAT,
/// inline assembly, and named metadata. Normalization must return all of it.
constexpr llvm::StringRef representativeModuleAssembly = R"(
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

source_filename = "loom_payload_anchor.c"

module asm "\09.globl loom_payload_anchor_marker"

$shared = comdat any

@counter = internal global i32 0, align 4
@shared_state = linkonce_odr global i32 7, comdat($shared), align 4
@exported = global i32 3, align 4

declare void @external_sink(ptr noundef) #1

define hidden i32 @accumulate(ptr noalias nocapture readonly %src, i64 %count) local_unnamed_addr #0 !dbg !9 {
entry:
  %value = load i32, ptr %src, align 4
  %scaled = mul nsw i32 %value, 3
  %prior = load i32, ptr @counter, align 4
  %total = add nsw i32 %prior, %scaled
  store i32 %total, ptr @counter, align 4
  call void @external_sink(ptr noundef @exported), !dbg !12
  ret i32 %total
}

define linkonce_odr void @shared_helper(ptr sret(i32) align 4 %out) #0 comdat($shared) {
entry:
  store i32 1, ptr %out, align 4
  ret void
}

attributes #0 = { nounwind uwtable "frame-pointer"="all" "target-cpu"="x86-64" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.dbg.cu = !{!5}
!loom.anchor = !{!8}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"loom payload anchor"}
!5 = distinct !DICompileUnit(language: DW_LANG_C11, file: !6, producer: "loom payload anchor", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!6 = !DIFile(filename: "loom_payload_anchor.c", directory: "/loom/anchor")
!7 = !{}
!8 = !{!"relocatable accelerator payload"}
!9 = distinct !DISubprogram(name: "accumulate", scope: !6, file: !6, line: 5, type: !10, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 7, column: 3, scope: !9)
)";

/// The same module written with equivalent target spellings: a triple missing
/// its vendor component and a reordered data layout with a redundant
/// three-component integer specification.
std::string equivalentSpellingAssembly() {
  std::string assembly = representativeModuleAssembly.str();
  const std::string canonicalTargets =
      "target datalayout = \"" + canonicalDataLayout.str() +
      "\"\ntarget triple = \"" + canonicalTargetTriple.str() + "\"";
  const std::string equivalentTargets =
      "target datalayout = \"e-S128-n8:16:32:64-f80:128-i128:128-i64:64:64-"
      "p272:64:64-p271:32:32-p270:32:32-m:e\"\n"
      "target triple = \"x86_64-linux-gnu\"";
  const std::size_t at = assembly.find(canonicalTargets);
  if (at == std::string::npos)
    fail(__func__, "canonical target lines not found");
  return assembly.replace(at, canonicalTargets.size(), equivalentTargets);
}

/// Builds source bitcode outside the production writer contract so the
/// normalizer is always given foreign input.
std::vector<std::uint8_t> buildSourceBitcode(const char *test,
                                             llvm::StringRef assembly,
                                             bool preserveUseListOrder) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(assembly, diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, "test module assembly did not parse: " + message);
  }
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::WriteBitcodeToFile(*module, stream, preserveUseListOrder);
  return std::vector<std::uint8_t>(buffer.begin(), buffer.end());
}

/// Re-parses normalized bytes and prints the complete module so a single
/// comparison covers every preserved construct.
std::string printBitcodeAsAssembly(const char *test,
                                   llvm::ArrayRef<std::uint8_t> bitcode) {
  llvm::LLVMContext context;
  const llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      "loom.payload.anchor");
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      llvm::parseBitcodeFile(buffer, context);
  if (!module)
    fail(test, llvm::toString(module.takeError()));
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  (*module)->print(stream, nullptr);
  return printed;
}

std::vector<std::uint8_t> toVector(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::vector<std::uint8_t> bytesOf(llvm::StringRef text) {
  return std::vector<std::uint8_t>(text.bytes_begin(), text.bytes_end());
}

/// A byte sequence that is the same size but not the same value, which is what
/// makes a stored projection stale without changing the encoding shape.
std::vector<std::uint8_t> flipped(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::uint8_t> changed = toVector(bytes);
  changed.back() ^= 0x01;
  return changed;
}

using RootField =
    llvm::ArrayRef<std::uint8_t> detail::RelocatablePayloadRoot::*;

/// Re-encodes production canonical bytes with exactly one root field replaced,
/// through the production root codec. The hostile input therefore restates no
/// field order, framing, or digest formula of its own.
std::vector<std::uint8_t> rootWithField(const char *test,
                                        llvm::ArrayRef<std::uint8_t> canonical,
                                        RootField field,
                                        llvm::ArrayRef<std::uint8_t> value) {
  detail::RelocatablePayloadRoot root =
      takeExpected(test, detail::decodeRelocatablePayloadRoot(canonical));
  root.*field = value;
  return detail::encodeRelocatablePayloadRoot(root);
}

ResolvedFrontendConfigView anchorView() {
  return projectResolvedFrontendConfigView(defaultResolvedConfig());
}

RelocatableAcceleratorPayload anchorPayload(const char *test) {
  return takeExpected(
      test, RelocatableAcceleratorPayload::create(
                buildSourceBitcode(test, representativeModuleAssembly, false),
                anchorView()));
}

void normalizationIsRepeatableAndCanonicalizesTargetSpellings() {
  const NormalizedLlvmModule normalized = takeExpected(
      __func__, normalizeLlvmModule(buildSourceBitcode(
                    __func__, representativeModuleAssembly, false)));
  require(__func__, normalized.canonicalTargetTriple == canonicalTargetTriple,
          "unexpected canonical target triple: " +
              normalized.canonicalTargetTriple);
  require(__func__, normalized.canonicalDataLayout == canonicalDataLayout,
          "unexpected canonical data layout: " +
              normalized.canonicalDataLayout);
  // Renormalizing the normalizer's own output reproduces it byte for byte.
  const NormalizedLlvmModule again =
      takeExpected(__func__, normalizeLlvmModule(normalized.bitcode));
  require(__func__, again.bitcode == normalized.bitcode,
          "repeated normalization was not byte identical");
  require(__func__, again.bitcodeDigest == normalized.bitcodeDigest,
          "repeated normalization changed the bitcode digest");

  // Equivalent target spellings are accepted and stored canonically.
  const NormalizedLlvmModule equivalent = takeExpected(
      __func__, normalizeLlvmModule(buildSourceBitcode(
                    __func__, equivalentSpellingAssembly(), false)));
  require(__func__, equivalent.canonicalTargetTriple == canonicalTargetTriple,
          "an equivalent triple spelling was not canonicalized: " +
              equivalent.canonicalTargetTriple);
  require(__func__, equivalent.canonicalDataLayout == canonicalDataLayout,
          "an equivalent data layout spelling was not canonicalized: " +
              equivalent.canonicalDataLayout);
  require(__func__, equivalent.bitcode == normalized.bitcode,
          "equivalent target spellings produced different normalized bytes");
}

void useListOrderIsTheOnlyDroppedDetail() {
  // The two modules differ only in the use-list order of one global.
  const std::string ordered = representativeModuleAssembly.str() +
                              "\nuselistorder ptr @counter, { 1, 0 }\n";
  const std::vector<std::uint8_t> preservedPlain =
      buildSourceBitcode(__func__, representativeModuleAssembly, true);
  const std::vector<std::uint8_t> preservedOrdered =
      buildSourceBitcode(__func__, ordered, true);
  require(__func__, preservedPlain != preservedOrdered,
          "the use-list order fixture does not actually differ");

  const NormalizedLlvmModule plain =
      takeExpected(__func__, normalizeLlvmModule(preservedPlain));
  const NormalizedLlvmModule reordered =
      takeExpected(__func__, normalizeLlvmModule(preservedOrdered));
  require(__func__, plain.bitcode == reordered.bitcode,
          "use-list order changed the normalized bytes");
}

void normalizationPreservesTheCompleteModule() {
  const NormalizedLlvmModule normalized = takeExpected(
      __func__, normalizeLlvmModule(buildSourceBitcode(
                    __func__, representativeModuleAssembly, false)));
  const std::string printed =
      printBitcodeAsAssembly(__func__, normalized.bitcode);

  // Nothing but the canonical target spellings may differ from the source, so
  // the source module printed through the same path is the exact expectation.
  const std::string expected = printBitcodeAsAssembly(
      __func__,
      buildSourceBitcode(__func__, representativeModuleAssembly, false));
  require(__func__, printed == expected,
          "normalization changed the module:\n" + printed);

  // One marker per class the contract names, so the anchor cannot quietly
  // degrade into an empty module.
  for (llvm::StringRef marker :
       {llvm::StringRef("!llvm.module.flags"), llvm::StringRef("sret(i32)"),
        llvm::StringRef("DICompileUnit"),
        llvm::StringRef("source_filename = \"loom_payload_anchor.c\"")})
    require(__func__, llvm::StringRef(printed).contains(marker),
            "the preserved module lost " + marker.str());
}

void payloadRoundTripsThroughCanonicalBytes() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  require(__func__,
          llvm::StringRef(payload.llvmProvider().repositoryIdentity) ==
              buildSelectedLlvmProvider().repositoryIdentity,
          "the payload provider is not the build-selected provider");
  require(__func__, payload.targetTriple() == canonicalTargetTriple,
          "unexpected payload target triple: " + payload.targetTriple().str());
  require(__func__, payload.dataLayout() == canonicalDataLayout,
          "unexpected payload data layout: " + payload.dataLayout().str());

  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const RelocatableAcceleratorPayload decoded =
      takeExpected(__func__, decodeRelocatableAcceleratorPayload(
                                 RelocatableAcceleratorPayload::artifactSchema,
                                 canonical.bytes()));
  require(__func__, decoded.identity() == payload.identity(),
          "the decoded payload identity changed");
  require(__func__, decoded.normalizedBitcode() == payload.normalizedBitcode(),
          "the decoded normalized module bytes changed");
  require(__func__,
          decoded.abiCompatibilityKey() == payload.abiCompatibilityKey(),
          "the decoded ABI compatibility key changed");
  require(__func__,
          decoded.canonicalSemanticBytes().bytes() == canonical.bytes(),
          "re-encoding the decoded payload changed the canonical bytes");

  // Creating the same payload twice is deterministic.
  require(__func__, anchorPayload(__func__).identity() == payload.identity(),
          "payload creation was not deterministic");

  // The ABI key stored in the payload is the production key over its raw
  // fields, not a value the payload authored on its own.
  AbiCompatibilityKeyInputs inputs;
  inputs.repositoryIdentity = payload.llvmProvider().repositoryIdentity;
  inputs.fullCommitIdentity = payload.llvmProvider().fullCommitIdentity;
  inputs.canonicalTargetTriple = payload.targetTriple();
  inputs.canonicalDataLayout = payload.dataLayout();
  inputs.viewSchemaDescriptorBytes =
      payload.frontendConfigView().schemaDescriptorBytes();
  inputs.viewCanonicalBytes = payload.frontendConfigView().canonicalViewBytes();
  require(__func__,
          payload.abiCompatibilityKey() == computeAbiCompatibilityKey(inputs),
          "the stored ABI compatibility key is not the production key");
}

void moduleContentChangesPayloadIdentity() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);

  std::string changedAssembly = representativeModuleAssembly.str();
  const std::size_t at = changedAssembly.find("i32 7, comdat");
  require(__func__, at != std::string::npos, "semantic fixture not found");
  changedAssembly.replace(at, std::strlen("i32 7"), "i32 9");

  const RelocatableAcceleratorPayload changed = takeExpected(
      __func__,
      RelocatableAcceleratorPayload::create(
          buildSourceBitcode(__func__, changedAssembly, false), anchorView()));
  require(__func__,
          changed.normalizedBitcodeDigest() !=
              payload.normalizedBitcodeDigest(),
          "a module content change did not change the bitcode digest");
  require(__func__, changed.identity() != payload.identity(),
          "a module content change did not change the payload identity");
  require(__func__,
          changed.abiCompatibilityKey() == payload.abiCompatibilityKey(),
          "a module content change moved the payload out of its ABI cohort");
}

/// Normalization fails closed rather than repairing an input: one target fact
/// the pinned provider does not print, and one unparsable module.
void normalizationFailsClosed() {
  const char *test = __func__;
  std::string foreignLayout = representativeModuleAssembly.str();
  const std::string layoutLine =
      "target datalayout = \"" + canonicalDataLayout.str() + "\"";
  const std::size_t at = foreignLayout.find(layoutLine);
  require(test, at != std::string::npos, "target fixture not found");
  foreignLayout.replace(
      at, layoutLine.size(),
      "target datalayout = \"e-m:e-p:64:64-i64:64-n8:16:32:64-S128\"");
  requireRejection(
      test,
      rejectionMessage(test, normalizeLlvmModule(buildSourceBitcode(
                                 test, foreignLayout, false))),
      "data_layout_not_canonical", "unexpected foreign data layout rejection");

  const std::vector<std::uint8_t> garbage(64, 0x5a);
  requireRejection(test, rejectionMessage(test, normalizeLlvmModule(garbage)),
                   "llvm_module_unparsable", "unexpected garbage rejection");
}

/// A payload the reader does not support, and canonical bytes that are not a
/// well-formed root, both fail closed before anything is interpreted.
void unsupportedSchemaAndMalformedEncodingAreRejected() {
  const char *test = __func__;
  const RelocatableAcceleratorPayload payload = anchorPayload(test);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();

  constexpr ArtifactSchemaDescriptor newerMajor{
      "loom.relocatable_accelerator_payload", SchemaVersion{2, 0}};
  requireRejection(test,
                   rejectionMessage(test, decodeRelocatableAcceleratorPayload(
                                              newerMajor, canonical.bytes())),
                   "relocatable_payload_schema_unsupported",
                   "unexpected schema rejection");

  const std::vector<std::uint8_t> truncated =
      toVector(canonical.bytes().drop_back(1));
  requireRejection(
      test,
      rejectionMessage(
          test, decodeRelocatableAcceleratorPayload(
                    RelocatableAcceleratorPayload::artifactSchema, truncated)),
      "relocatable_payload_encoding", "unexpected truncation rejection");
}

/// Every stored projection is recomputed from the raw fields, so a stale one is
/// a typed rejection. One case per independent recomputation the reader makes.
void staleRootProjectionsAreRejected() {
  const char *test = __func__;
  const RelocatableAcceleratorPayload payload = anchorPayload(test);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const ComponentViewDigest viewDigest = payload.frontendConfigView().digest();
  using Root = detail::RelocatablePayloadRoot;

  struct Case {
    const char *what;
    RootField field;
    std::vector<std::uint8_t> value;
    llvm::StringRef marker;
  };
  const Case cases[] = {
      {"provider identity", &Root::repositoryIdentity,
       bytesOf("not-the-pinned-provider"), "llvm_provider_mismatch"},
      {"frontend view digest", &Root::viewDigest, flipped(viewDigest.bytes()),
       "component_view_digest_mismatch"},
      {"target triple", &Root::targetTriple,
       bytesOf("aarch64-unknown-linux-gnu"), "target_triple_mismatch"},
      {"normalized bitcode digest", &Root::normalizedBitcodeDigest,
       flipped(payload.normalizedBitcodeDigest()),
       "normalized_bitcode_digest_mismatch"},
      {"abi compatibility key", &Root::abiCompatibilityKey,
       flipped(payload.abiCompatibilityKey().bytes()),
       "abi_compatibility_key_mismatch"},
  };
  for (const Case &testCase : cases)
    requireRejection(
        test,
        rejectionMessage(test,
                         decodeRelocatableAcceleratorPayload(
                             RelocatableAcceleratorPayload::artifactSchema,
                             rootWithField(test, canonical.bytes(),
                                           testCase.field, testCase.value))),
        testCase.marker,
        std::string("unexpected rejection for a stale ") + testCase.what);
}

/// Valid bitcode of the same module that the production writer contract would
/// never emit is rejected: the reader rewrites the module through that writer
/// and requires exact byte equality before it trusts anything derived from it.
void noncanonicalModuleBytesAreRejected() {
  const char *test = __func__;
  const RelocatableAcceleratorPayload payload = anchorPayload(test);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();

  const std::vector<std::uint8_t> noncanonical =
      buildSourceBitcode(test, representativeModuleAssembly, true);
  require(test, noncanonical != toVector(payload.normalizedBitcode()),
          "the noncanonical fixture matches the production writer output");
  requireRejection(
      test,
      rejectionMessage(
          test,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema,
              rootWithField(test, canonical.bytes(),
                            &detail::RelocatablePayloadRoot::normalizedBitcode,
                            noncanonical))),
      "normalized_bitcode_not_canonical",
      "unexpected noncanonical module rejection");
}

} // namespace

int main() {
  normalizationIsRepeatableAndCanonicalizesTargetSpellings();
  useListOrderIsTheOnlyDroppedDetail();
  normalizationPreservesTheCompleteModule();
  payloadRoundTripsThroughCanonicalBytes();
  moduleContentChangesPayloadIdentity();
  normalizationFailsClosed();
  unsupportedSchemaAndMalformedEncodingAreRejected();
  staleRootProjectionsAreRejected();
  noncanonicalModuleBytesAreRejected();
  return 0;
}
