#include "Frontend/Raising/CandidateHints.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<CandidateHintError>(
      CandidateHintErrorKind::InvalidEncoding, message.str());
}

llvm::Error validate(const FunctionCandidateAnnotation &hint) {
  if (hint.sourceFile.empty())
    return invalid("source file is empty");
  if (hint.sourceFile.find('\0') != std::string::npos)
    return invalid("source file contains a null byte");
  if (hint.sourceFile.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("source file is too long");
  if (hint.carrier.line == 0 || hint.carrier.column == 0 ||
      hint.pragma.line == 0 || hint.pragma.column == 0 ||
      hint.targetBegin.line == 0 || hint.targetBegin.column == 0 ||
      hint.targetEnd.line == 0 || hint.targetEnd.column == 0)
    return invalid("source positions must be one-based");
  // These are presumed (possibly #line-remapped) coordinates. Translation
  // unit ordering is checked by the Clang producer while the coordinates are
  // retained as source metadata; comparing them here would reject legal
  // remapped files.
  return llvm::Error::success();
}

llvm::Error validate(const LoopCandidateAnnotation &hint) {
  if (hint.marker == 0)
    return invalid("loop marker must be nonzero");
  if (hint.sourceFile.empty())
    return invalid("source file is empty");
  if (hint.sourceFile.find('\0') != std::string::npos)
    return invalid("source file contains a null byte");
  if (hint.sourceFile.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("source file is too long");
  if (hint.carrier.line == 0 || hint.carrier.column == 0 ||
      hint.pragma.line == 0 || hint.pragma.column == 0 ||
      hint.targetBegin.line == 0 || hint.targetBegin.column == 0 ||
      hint.targetEnd.line == 0 || hint.targetEnd.column == 0)
    return invalid("source positions must be one-based");
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> takeUnsigned(llvm::StringRef &suffix,
                                           std::uint64_t maximum,
                                           llvm::StringRef typeName) {
  const std::size_t separator = suffix.find('|');
  llvm::StringRef field = separator == llvm::StringRef::npos
                              ? suffix
                              : suffix.take_front(separator);
  if (field.empty())
    return invalid("numeric field is empty");
  std::uint64_t value = 0;
  for (char character : field) {
    if (character < '0' || character > '9')
      return invalid("numeric field is not decimal");
    auto shifted = llvm::checkedMulUnsigned(value, std::uint64_t{10});
    if (!shifted)
      return invalid("numeric field overflows u64");
    auto extended = llvm::checkedAddUnsigned(
        *shifted, static_cast<std::uint64_t>(character - '0'));
    if (!extended || *extended > maximum)
      return invalid(llvm::Twine("numeric field overflows ") + typeName);
    value = *extended;
  }
  suffix = separator == llvm::StringRef::npos
               ? llvm::StringRef{}
               : suffix.drop_front(separator + 1);
  return value;
}

llvm::Expected<std::uint32_t> takeU32(llvm::StringRef &suffix) {
  auto value =
      takeUnsigned(suffix, std::numeric_limits<std::uint32_t>::max(), "u32");
  if (!value)
    return value.takeError();
  return static_cast<std::uint32_t>(*value);
}

llvm::Expected<std::uint64_t> takeU64(llvm::StringRef &suffix) {
  return takeUnsigned(suffix, std::numeric_limits<std::uint64_t>::max(), "u64");
}

void appendPosition(std::string &encoded, SourcePosition position) {
  encoded.push_back('|');
  encoded += std::to_string(position.line);
  encoded.push_back('|');
  encoded += std::to_string(position.column);
}

llvm::Expected<std::string> takeSourceFile(llvm::StringRef &suffix) {
  auto sourceFileSize = takeU32(suffix);
  if (!sourceFileSize)
    return sourceFileSize.takeError();
  if (suffix.size() <= *sourceFileSize || suffix[*sourceFileSize] != '|')
    return invalid("source file field is truncated");
  std::string sourceFile = suffix.take_front(*sourceFileSize).str();
  suffix = suffix.drop_front(*sourceFileSize + 1);
  return sourceFile;
}

llvm::Expected<SourcePosition> takePosition(llvm::StringRef &suffix) {
  auto line = takeU32(suffix);
  if (!line)
    return line.takeError();
  auto column = takeU32(suffix);
  if (!column)
    return column.takeError();
  return SourcePosition{*line, *column};
}

void appendSourceFile(std::string &encoded, llvm::StringRef sourceFile) {
  encoded.push_back('|');
  encoded += std::to_string(sourceFile.size());
  encoded.push_back('|');
  encoded += sourceFile;
}

} // namespace

char CandidateHintError::ID = 0;

llvm::StringRef candidateHintErrorKindName(CandidateHintErrorKind kind) {
  switch (kind) {
  case CandidateHintErrorKind::InvalidEncoding:
    return "invalid_encoding";
  case CandidateHintErrorKind::InvalidPlacement:
    return "invalid_placement";
  case CandidateHintErrorKind::UnsupportedConstruct:
    return "unsupported_construct";
  case CandidateHintErrorKind::ProjectionProofNotEstablished:
    return "projection_proof_not_established";
  }
  llvm_unreachable("unknown candidate hint error kind");
}

void CandidateHintError::log(llvm::raw_ostream &stream) const {
  stream << "candidate_hint_" << candidateHintErrorKindName(kind_) << ": "
         << message_;
}

std::error_code CandidateHintError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<std::string>
encodeFunctionCandidateAnnotation(const FunctionCandidateAnnotation &hint) {
  if (llvm::Error error = validate(hint))
    return std::move(error);
  std::string encoded = functionCandidateAnnotationSchema.str();
  appendSourceFile(encoded, hint.sourceFile);
  appendPosition(encoded, hint.carrier);
  appendPosition(encoded, hint.pragma);
  appendPosition(encoded, hint.targetBegin);
  appendPosition(encoded, hint.targetEnd);
  return encoded;
}

llvm::Expected<FunctionCandidateAnnotation>
decodeFunctionCandidateAnnotation(llvm::StringRef annotation) {
  llvm::StringRef suffix = annotation;
  if (!suffix.consume_front(functionCandidateAnnotationSchema) ||
      !suffix.consume_front("|"))
    return invalid("annotation has an unsupported schema");

  FunctionCandidateAnnotation hint;
  auto sourceFile = takeSourceFile(suffix);
  if (!sourceFile)
    return sourceFile.takeError();
  hint.sourceFile = std::move(*sourceFile);
  auto carrier = takePosition(suffix);
  if (!carrier)
    return carrier.takeError();
  auto pragma = takePosition(suffix);
  if (!pragma)
    return pragma.takeError();
  auto targetBegin = takePosition(suffix);
  if (!targetBegin)
    return targetBegin.takeError();
  auto targetEnd = takePosition(suffix);
  if (!targetEnd)
    return targetEnd.takeError();
  if (!suffix.empty())
    return invalid("annotation has trailing fields");
  hint.carrier = *carrier;
  hint.pragma = *pragma;
  hint.targetBegin = *targetBegin;
  hint.targetEnd = *targetEnd;
  if (llvm::Error error = validate(hint))
    return std::move(error);
  auto reencoded = encodeFunctionCandidateAnnotation(hint);
  if (!reencoded)
    return reencoded.takeError();
  if (*reencoded != annotation)
    return invalid("annotation does not re-encode exactly");
  return hint;
}

llvm::Expected<std::string>
encodeLoopCandidateAnnotation(const LoopCandidateAnnotation &hint) {
  if (llvm::Error error = validate(hint))
    return std::move(error);
  std::string encoded = loopCandidateAnnotationSchema.str();
  encoded.push_back('|');
  encoded += std::to_string(hint.marker);
  appendSourceFile(encoded, hint.sourceFile);
  appendPosition(encoded, hint.carrier);
  appendPosition(encoded, hint.pragma);
  appendPosition(encoded, hint.targetBegin);
  appendPosition(encoded, hint.targetEnd);
  return encoded;
}

llvm::Expected<LoopCandidateAnnotation>
decodeLoopCandidateAnnotation(llvm::StringRef annotation) {
  llvm::StringRef suffix = annotation;
  if (!suffix.consume_front(loopCandidateAnnotationSchema) ||
      !suffix.consume_front("|"))
    return invalid("annotation has an unsupported schema");

  LoopCandidateAnnotation hint;
  auto marker = takeU64(suffix);
  if (!marker)
    return marker.takeError();
  hint.marker = *marker;
  auto sourceFile = takeSourceFile(suffix);
  if (!sourceFile)
    return sourceFile.takeError();
  hint.sourceFile = std::move(*sourceFile);
  auto carrier = takePosition(suffix);
  if (!carrier)
    return carrier.takeError();
  auto pragma = takePosition(suffix);
  if (!pragma)
    return pragma.takeError();
  auto targetBegin = takePosition(suffix);
  if (!targetBegin)
    return targetBegin.takeError();
  auto targetEnd = takePosition(suffix);
  if (!targetEnd)
    return targetEnd.takeError();
  if (!suffix.empty())
    return invalid("annotation has trailing fields");
  hint.carrier = *carrier;
  hint.pragma = *pragma;
  hint.targetBegin = *targetBegin;
  hint.targetEnd = *targetEnd;
  if (llvm::Error error = validate(hint))
    return std::move(error);
  auto reencoded = encodeLoopCandidateAnnotation(hint);
  if (!reencoded)
    return reencoded.takeError();
  if (*reencoded != annotation)
    return invalid("annotation does not re-encode exactly");
  return hint;
}

std::string encodeLoopCandidateMarker(std::uint64_t marker) {
  std::string encoded = loopCandidateMarkerSchema.str();
  encoded.push_back('|');
  encoded += std::to_string(marker);
  return encoded;
}

llvm::Expected<std::uint64_t>
decodeLoopCandidateMarker(llvm::StringRef annotation) {
  llvm::StringRef suffix = annotation;
  if (!suffix.consume_front(loopCandidateMarkerSchema) ||
      !suffix.consume_front("|"))
    return invalid("loop marker has an unsupported schema");
  auto marker = takeU64(suffix);
  if (!marker)
    return marker.takeError();
  if (*marker == 0)
    return invalid("loop marker must be nonzero");
  if (!suffix.empty())
    return invalid("loop marker has trailing fields");
  if (encodeLoopCandidateMarker(*marker) != annotation)
    return invalid("loop marker does not re-encode exactly");
  return *marker;
}

llvm::Error removeCandidateTemporaryRetention(llvm::Module &module) {
  llvm::SmallPtrSet<llvm::GlobalValue *, 8> projected;
  for (llvm::Function &function : module)
    if (function.getMetadata(candidateTemporaryRetentionMetadataName))
      projected.insert(&function);
  if (projected.empty())
    return llvm::Error::success();

  llvm::GlobalVariable *used = module.getNamedGlobal("llvm.compiler.used");
  if (!used || !used->hasInitializer())
    return llvm::make_error<CandidateHintError>(
        CandidateHintErrorKind::ProjectionProofNotEstablished,
        "temporary candidate retention lost llvm.compiler.used");
  auto *initializer =
      llvm::dyn_cast<llvm::ConstantArray>(used->getInitializer());
  if (!initializer || !used->use_empty())
    return llvm::make_error<CandidateHintError>(
        CandidateHintErrorKind::ProjectionProofNotEstablished,
        "temporary candidate retention has a malformed owner");

  llvm::SmallPtrSet<llvm::GlobalValue *, 8> found;
  llvm::SmallVector<llvm::Constant *> retained;
  for (llvm::Value *operand : initializer->operand_values()) {
    auto *constant = llvm::dyn_cast<llvm::Constant>(operand);
    if (!constant)
      return llvm::make_error<CandidateHintError>(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "temporary candidate retention contains a non-constant entry");
    auto *global =
        llvm::dyn_cast<llvm::GlobalValue>(constant->stripPointerCasts());
    if (!global || !projected.contains(global)) {
      retained.push_back(constant);
      continue;
    }
    if (!found.insert(global).second)
      return llvm::make_error<CandidateHintError>(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "temporary candidate retention contains a duplicate entry");
  }
  if (found.size() != projected.size())
    return llvm::make_error<CandidateHintError>(
        CandidateHintErrorKind::ProjectionProofNotEstablished,
        "temporary candidate retention lost a projected carrier");

  for (llvm::GlobalValue *global : projected)
    llvm::cast<llvm::Function>(global)->setMetadata(
        candidateTemporaryRetentionMetadataName, nullptr);
  if (!retained.empty()) {
    auto *elementType =
        llvm::cast<llvm::ArrayType>(initializer->getType())->getElementType();
    auto *arrayType = llvm::ArrayType::get(elementType, retained.size());
    auto *replacement = new llvm::GlobalVariable(
        module, arrayType, false, llvm::GlobalValue::AppendingLinkage,
        llvm::ConstantArray::get(arrayType, retained), "", used,
        used->getThreadLocalMode(), used->getAddressSpace());
    replacement->copyAttributesFrom(used);
    replacement->takeName(used);
  }
  used->eraseFromParent();
  return llvm::Error::success();
}

} // namespace loom::raising
