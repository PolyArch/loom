#include "Frontend/Raising/CandidateHints.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "candidate_hint_invalid: " + message);
}

llvm::Error validate(const FunctionCandidateAnnotation &hint) {
  if (hint.sourceFile.empty())
    return invalid("source file is empty");
  if (hint.sourceFile.find('\0') != std::string::npos)
    return invalid("source file contains a null byte");
  if (hint.sourceFile.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("source file is too long");
  if (hint.pragma.line == 0 || hint.pragma.column == 0 ||
      hint.targetBegin.line == 0 || hint.targetBegin.column == 0 ||
      hint.targetEnd.line == 0 || hint.targetEnd.column == 0)
    return invalid("source positions must be one-based");
  // These are presumed (possibly #line-remapped) coordinates. Translation
  // unit ordering is checked by the Clang producer while the coordinates are
  // retained as source metadata; comparing them here would reject legal
  // remapped files.
  return llvm::Error::success();
}

llvm::Expected<std::uint32_t> takeU32(llvm::StringRef &suffix) {
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
    if (!extended || *extended > std::numeric_limits<std::uint32_t>::max())
      return invalid("numeric field overflows u32");
    value = *extended;
  }
  suffix = separator == llvm::StringRef::npos
               ? llvm::StringRef{}
               : suffix.drop_front(separator + 1);
  return static_cast<std::uint32_t>(value);
}

void appendPosition(std::string &encoded, SourcePosition position) {
  encoded.push_back('|');
  encoded += std::to_string(position.line);
  encoded.push_back('|');
  encoded += std::to_string(position.column);
}

} // namespace

llvm::Expected<std::string>
encodeFunctionCandidateAnnotation(const FunctionCandidateAnnotation &hint) {
  if (llvm::Error error = validate(hint))
    return std::move(error);
  std::string encoded = functionCandidateAnnotationSchema.str();
  encoded.push_back('|');
  encoded += std::to_string(hint.sourceFile.size());
  encoded.push_back('|');
  encoded += hint.sourceFile;
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

  auto sourceFileSize = takeU32(suffix);
  if (!sourceFileSize)
    return sourceFileSize.takeError();
  if (suffix.size() <= *sourceFileSize || suffix[*sourceFileSize] != '|')
    return invalid("source file field is truncated");
  FunctionCandidateAnnotation hint;
  hint.sourceFile = suffix.take_front(*sourceFileSize).str();
  suffix = suffix.drop_front(*sourceFileSize + 1);

  auto takePosition = [&]() -> llvm::Expected<SourcePosition> {
    auto line = takeU32(suffix);
    if (!line)
      return line.takeError();
    auto column = takeU32(suffix);
    if (!column)
      return column.takeError();
    return SourcePosition{*line, *column};
  };
  auto pragma = takePosition();
  if (!pragma)
    return pragma.takeError();
  auto targetBegin = takePosition();
  if (!targetBegin)
    return targetBegin.takeError();
  auto targetEnd = takePosition();
  if (!targetEnd)
    return targetEnd.takeError();
  if (!suffix.empty())
    return invalid("annotation has trailing fields");
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

} // namespace loom::raising
