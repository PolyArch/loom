#ifndef LOOM_LIB_EVALUATION_CANONICALSUPPORT_H
#define LOOM_LIB_EVALUATION_CANONICALSUPPORT_H

#include "Evaluation/Case.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

// Shared canonical-byte framing and strict JSON decoding used by every
// Evaluation value schema. Framing is big-endian with explicit lengths so no
// key is a prefix of another. Cross-artifact reference framing itself is
// owned by Common; these helpers only compose it.

namespace loom::evaluation::detail {

llvm::Error evaluationError(const llvm::Twine &message);

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value);
void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value);
void appendI64Be(std::vector<std::uint8_t> &bytes, std::int64_t value);
void appendFramedBytes(std::vector<std::uint8_t> &bytes,
                       llvm::ArrayRef<std::uint8_t> payload);
void appendFramedString(std::vector<std::uint8_t> &bytes, llvm::StringRef text);
void appendSchemaVersion(std::vector<std::uint8_t> &bytes,
                         SchemaVersion version);
void appendDecimalValue(std::vector<std::uint8_t> &bytes, DecimalValue value);
void appendExactRatio(std::vector<std::uint8_t> &bytes, ExactRatio value);

/// Framed target key shared by the canonical scope key and every condition
/// key that carries a target. The reference framing is exactly the Common
/// heterogeneous framing.
void appendSubjectTargetKey(std::vector<std::uint8_t> &bytes,
                            const SubjectTargetRef &target);

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                std::initializer_list<llvm::StringRef> allowed);
llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context);
llvm::Expected<std::int64_t> requireInteger(const llvm::json::Object &object,
                                            llvm::StringRef key,
                                            llvm::StringRef context);
llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context);
llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context);
llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context);

} // namespace loom::evaluation::detail

#endif // LOOM_LIB_EVALUATION_CANONICALSUPPORT_H
