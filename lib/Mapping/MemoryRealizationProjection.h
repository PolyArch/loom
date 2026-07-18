#ifndef LOOM_LIB_MAPPING_MEMORYREALIZATIONPROJECTION_H
#define LOOM_LIB_MAPPING_MEMORYREALIZATIONPROJECTION_H

#include "FabricOccurrenceIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <vector>

namespace loom::mapping::detail {

llvm::Expected<std::vector<ValidatedMemoryRealizationProjection>>
buildMemoryRealizationProjections(
    llvm::ArrayRef<MemoryRealizationDraft> realizations,
    const std::map<std::uint64_t, const MemorySemanticEncodingDescriptor *>
        &encodings,
    const std::map<std::uint64_t, const MemoryImplementationDescriptor *>
        &implementations,
    const std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
        &operationTemplates);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_MEMORYREALIZATIONPROJECTION_H
