#ifndef LOOM_LIB_FRONTEND_RAISING_PRESERVEDHINTS_H
#define LOOM_LIB_FRONTEND_RAISING_PRESERVEDHINTS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace loom {
namespace raising {

// Carrier under which an imported LLVM loop annotation stays associated with
// the loop it describes once mechanical raising has respelled the branch that
// carried it. The value is the imported LLVM::LoopAnnotationAttr itself; only
// its carrier changes, from the latch branch to the cf branch that replaces it
// and then to the structured loop that owns the recovered cycle. A raising
// step that cannot move it to the operation owning the same loop preserves the
// construct it describes instead of dropping it.
inline constexpr ::llvm::StringLiteral loopAnnotationName =
    "llvm.loop_annotation";

// Attach `annotation`, if any, to the operation that now owns the branch or
// loop it describes.
inline void carryLoopAnnotation(::mlir::Attribute annotation,
                                ::mlir::Operation *owner) {
  if (annotation)
    owner->setAttr(loopAnnotationName, annotation);
}

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_PRESERVEDHINTS_H
