#include "Frontend/Payload/RelocatableAcceleratorPayload.h"
#include "Common/Artifact.h"
#include "Common/ResolvedConfig.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/LlvmModuleNormalization.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

std::vector<std::uint8_t> sha256(llvm::ArrayRef<std::uint8_t> bytes) {
  const std::array<std::uint8_t, 32> digest = llvm::SHA256::hash(bytes);
  return std::vector<std::uint8_t>(digest.begin(), digest.end());
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::vector<std::uint8_t> toVector(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

/// Flips one byte of the named root field inside production canonical bytes.
/// The search stops before the trailing normalized module bytes, so exactly one
/// framed root field changes.
std::vector<std::uint8_t>
mutateRootField(const char *test, const char *what,
                const CanonicalSemanticBytes &canonical,
                llvm::ArrayRef<std::uint8_t> moduleBytes,
                llvm::ArrayRef<std::uint8_t> fieldBytes) {
  std::vector<std::uint8_t> mutated = toVector(canonical.bytes());
  const std::size_t rootHeaderSize = mutated.size() - moduleBytes.size();
  const auto begin = mutated.begin();
  const auto end = begin + static_cast<std::ptrdiff_t>(rootHeaderSize);
  const auto at = std::search(begin, end, fieldBytes.begin(), fieldBytes.end());
  if (at == end)
    fail(test, std::string(what) + " was not found in the canonical bytes");
  if (std::search(at + 1, end, fieldBytes.begin(), fieldBytes.end()) != end)
    fail(test, std::string(what) + " is ambiguous in the canonical bytes");
  *(at + static_cast<std::ptrdiff_t>(fieldBytes.size()) - 1) ^= 0x01;
  return mutated;
}

/// Rebuilds only the trailing normalized-bitcode digest and bytes so the
/// reader's canonical-byte check is what the hostile input exercises.
std::vector<std::uint8_t>
withReplacedModuleBytes(const CanonicalSemanticBytes &canonical,
                        llvm::ArrayRef<std::uint8_t> moduleBytes,
                        llvm::ArrayRef<std::uint8_t> replacement) {
  const llvm::ArrayRef<std::uint8_t> bytes = canonical.bytes();
  std::vector<std::uint8_t> rebuilt =
      toVector(bytes.drop_back(32 + 8 + moduleBytes.size()));
  const std::vector<std::uint8_t> digest = sha256(replacement);
  rebuilt.insert(rebuilt.end(), digest.begin(), digest.end());
  appendU64Be(rebuilt, replacement.size());
  rebuilt.insert(rebuilt.end(), replacement.begin(), replacement.end());
  return rebuilt;
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
  require(__func__,
          normalized.bitcodeDigest == llvm::SHA256::hash(normalized.bitcode),
          "the stored digest is not the digest of the normalized bytes");

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

  // Guard against the anchor quietly degrading into an empty module.
  for (llvm::StringRef marker :
       {llvm::StringRef("source_filename = \"loom_payload_anchor.c\""),
        llvm::StringRef("module asm"), llvm::StringRef("comdat any"),
        llvm::StringRef("linkonce_odr"), llvm::StringRef("hidden"),
        llvm::StringRef("internal global"), llvm::StringRef("sret(i32)"),
        llvm::StringRef("noalias"),
        llvm::StringRef("\"frame-pointer\"=\"all\""),
        llvm::StringRef("!llvm.module.flags"), llvm::StringRef("!llvm.dbg.cu"),
        llvm::StringRef("!loom.anchor"), llvm::StringRef("DICompileUnit"),
        llvm::StringRef("DILocation"), llvm::StringRef("Debug Info Version")})
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

void invalidTargetFactsAreRejected() {
  struct Case {
    const char *what;
    std::string search;
    std::string replacement;
    llvm::StringRef marker;
  };
  const std::string tripleLine =
      "target triple = \"" + canonicalTargetTriple.str() + "\"";
  const std::string layoutLine =
      "target datalayout = \"" + canonicalDataLayout.str() + "\"";
  const Case cases[] = {
      {"absent target triple", tripleLine, "", "target_triple_absent"},
      {"absent data layout", layoutLine, "", "data_layout_absent"},
      {"unsupported target triple", tripleLine,
       "target triple = \"not-a-real-triple\"", "target_triple_unsupported"},
      {"foreign data layout", layoutLine,
       "target datalayout = \"e-m:e-p:64:64-i64:64-n8:16:32:64-S128\"",
       "data_layout_not_canonical"},
  };
  for (const Case &testCase : cases) {
    std::string assembly = representativeModuleAssembly.str();
    const std::size_t at = assembly.find(testCase.search);
    require(__func__, at != std::string::npos,
            std::string("target fixture not found for ") + testCase.what);
    assembly.replace(at, testCase.search.size(), testCase.replacement);
    requireRejection(
        __func__,
        rejectionMessage(__func__, normalizeLlvmModule(buildSourceBitcode(
                                       __func__, assembly, false))),
        testCase.marker,
        std::string("unexpected rejection for ") + testCase.what);
  }
}

void malformedBitcodeIsRejected() {
  const std::vector<std::uint8_t> garbage(64, 0x5a);
  requireRejection(__func__,
                   rejectionMessage(__func__, normalizeLlvmModule(garbage)),
                   "llvm_module_unparsable", "unexpected garbage rejection");
  requireRejection(__func__,
                   rejectionMessage(__func__, normalizeLlvmModule({})),
                   "llvm_module_unparsable", "unexpected empty rejection");

  const std::vector<std::uint8_t> valid =
      buildSourceBitcode(__func__, representativeModuleAssembly, false);
  const std::vector<std::uint8_t> truncated(valid.begin(),
                                            valid.begin() + valid.size() / 2);
  requireRejection(__func__,
                   rejectionMessage(__func__, normalizeLlvmModule(truncated)),
                   "llvm_module_unparsable", "unexpected truncation rejection");
}

void unsupportedSchemaIsRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();

  constexpr ArtifactSchemaDescriptor foreignIdentity{"loom.other.payload",
                                                     SchemaVersion{1, 0}};
  constexpr ArtifactSchemaDescriptor newerMajor{
      "loom.relocatable_accelerator_payload", SchemaVersion{2, 0}};
  constexpr ArtifactSchemaDescriptor newerMinor{
      "loom.relocatable_accelerator_payload", SchemaVersion{1, 1}};
  for (const ArtifactSchemaDescriptor &schema :
       {foreignIdentity, newerMajor, newerMinor})
    requireRejection(
        __func__,
        rejectionMessage(__func__, decodeRelocatableAcceleratorPayload(
                                       schema, canonical.bytes())),
        "relocatable_payload_schema_unsupported",
        "unexpected schema rejection");
}

void malformedEncodingsAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const llvm::ArrayRef<std::uint8_t> bytes = canonical.bytes();

  const std::vector<std::uint8_t> truncated = toVector(bytes.drop_back(1));
  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema, truncated)),
      "relocatable_payload_encoding", "unexpected truncation rejection");

  std::vector<std::uint8_t> trailing = toVector(bytes);
  trailing.push_back(0x00);
  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema, trailing)),
      "relocatable_payload_encoding", "unexpected trailing byte rejection");

  // The leading framed length claims far more bytes than the input holds.
  std::vector<std::uint8_t> overflow = toVector(bytes);
  for (unsigned index = 0; index < 8; ++index)
    overflow[index] = 0xff;
  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema, overflow)),
      "relocatable_payload_encoding", "unexpected length overflow rejection");

  requireRejection(
      __func__,
      rejectionMessage(__func__,
                       decodeRelocatableAcceleratorPayload(
                           RelocatableAcceleratorPayload::artifactSchema, {})),
      "relocatable_payload_encoding", "unexpected empty encoding rejection");
}

void staleRootFieldsAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const llvm::ArrayRef<std::uint8_t> moduleBytes = payload.normalizedBitcode();

  auto asBytes = [](llvm::StringRef text) {
    return llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
  };

  // The view digest is derived on demand, so it is held here rather than
  // referenced through a temporary.
  const ComponentViewDigest viewDigest = payload.frontendConfigView().digest();

  struct Case {
    const char *what;
    llvm::ArrayRef<std::uint8_t> field;
    llvm::StringRef marker;
  };
  const Case cases[] = {
      {"provider repository identity",
       asBytes(payload.llvmProvider().repositoryIdentity),
       "llvm_provider_mismatch"},
      {"provider commit identity",
       asBytes(payload.llvmProvider().fullCommitIdentity),
       "llvm_provider_mismatch"},
      {"frontend view descriptor",
       payload.frontendConfigView().schemaDescriptorBytes(),
       "frontend_config_view_descriptor"},
      {"frontend view digest", viewDigest.bytes(),
       "component_view_digest_mismatch"},
      {"normalized bitcode digest", payload.normalizedBitcodeDigest(),
       "normalized_bitcode_digest_mismatch"},
      {"target triple", asBytes(payload.targetTriple()),
       "target_triple_mismatch"},
      {"data layout", asBytes(payload.dataLayout()), "data_layout_mismatch"},
      {"abi compatibility key", payload.abiCompatibilityKey().bytes(),
       "abi_compatibility_key_mismatch"},
  };
  for (const Case &testCase : cases)
    requireRejection(
        __func__,
        rejectionMessage(__func__,
                         decodeRelocatableAcceleratorPayload(
                             RelocatableAcceleratorPayload::artifactSchema,
                             mutateRootField(__func__, testCase.what, canonical,
                                             moduleBytes, testCase.field))),
        testCase.marker,
        std::string("unexpected rejection for a stale ") + testCase.what);
}

void noncanonicalModuleBytesAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();

  // Valid bitcode of the same module that the production writer contract would
  // never emit, carried with a matching digest so only canonicality is tested.
  const std::vector<std::uint8_t> noncanonical =
      buildSourceBitcode(__func__, representativeModuleAssembly, true);
  require(__func__, noncanonical != toVector(payload.normalizedBitcode()),
          "the noncanonical fixture matches the production writer output");
  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema,
              withReplacedModuleBytes(canonical, payload.normalizedBitcode(),
                                      noncanonical))),
      "normalized_bitcode_not_canonical",
      "unexpected noncanonical module rejection");

  const std::vector<std::uint8_t> garbage(64, 0x5a);
  requireRejection(
      __func__,
      rejectionMessage(
          __func__, decodeRelocatableAcceleratorPayload(
                        RelocatableAcceleratorPayload::artifactSchema,
                        withReplacedModuleBytes(
                            canonical, payload.normalizedBitcode(), garbage))),
      "llvm_module_unparsable", "unexpected malformed module rejection");
}

void compatibilityRequiresExactRawFields() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);

  std::string otherAssembly = representativeModuleAssembly.str();
  const std::size_t at = otherAssembly.find("i32 7, comdat");
  require(__func__, at != std::string::npos, "compatibility fixture not found");
  otherAssembly.replace(at, std::strlen("i32 7"), "i32 11");
  const RelocatableAcceleratorPayload sibling = takeExpected(
      __func__,
      RelocatableAcceleratorPayload::create(
          buildSourceBitcode(__func__, otherAssembly, false), anchorView()));

  // Distinct translation units of one cohort stay compatible.
  require(__func__, sibling.identity() != payload.identity(),
          "the compatibility fixture is not a distinct payload");
  require(
      __func__,
      llvm::toString(requireRelocatablePayloadCompatibility(payload, sibling))
          .empty(),
      "two payloads with identical raw fields were rejected");
  require(
      __func__,
      llvm::toString(requireRelocatablePayloadCompatibility(payload, payload))
          .empty(),
      "a payload was incompatible with itself");

  // One raw field disagreement is a typed rejection.
  std::string aarch64Assembly = representativeModuleAssembly.str();
  const std::string tripleLine =
      "target triple = \"" + canonicalTargetTriple.str() + "\"";
  const std::string layoutLine =
      "target datalayout = \"" + canonicalDataLayout.str() + "\"";
  const std::size_t tripleAt = aarch64Assembly.find(tripleLine);
  aarch64Assembly.replace(tripleAt, tripleLine.size(),
                          "target triple = \"aarch64-unknown-linux-gnu\"");
  const std::size_t layoutAt = aarch64Assembly.find(layoutLine);
  aarch64Assembly.replace(
      layoutAt, layoutLine.size(),
      "target datalayout = \"" +
          llvm::Triple("aarch64-unknown-linux-gnu").computeDataLayout() + "\"");
  const RelocatableAcceleratorPayload foreign = takeExpected(
      __func__,
      RelocatableAcceleratorPayload::create(
          buildSourceBitcode(__func__, aarch64Assembly, false), anchorView()));
  requireRejection(
      __func__,
      llvm::toString(requireRelocatablePayloadCompatibility(payload, foreign)),
      "relocatable_payload_incompatible",
      "a foreign target payload was accepted as compatible");
}

} // namespace

int main() {
  normalizationIsRepeatableAndCanonicalizesTargetSpellings();
  useListOrderIsTheOnlyDroppedDetail();
  normalizationPreservesTheCompleteModule();
  payloadRoundTripsThroughCanonicalBytes();
  moduleContentChangesPayloadIdentity();
  invalidTargetFactsAreRejected();
  malformedBitcodeIsRejected();
  unsupportedSchemaIsRejected();
  malformedEncodingsAreRejected();
  staleRootFieldsAreRejected();
  noncanonicalModuleBytesAreRejected();
  compatibilityRequiresExactRawFields();
  return 0;
}
