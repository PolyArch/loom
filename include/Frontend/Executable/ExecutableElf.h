#ifndef LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H
#define LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H

#include "llvm/Support/Error.h"

namespace llvm::object {
class ELFObjectFileBase;
}

namespace loom {

class CompilerTargetBinding;

llvm::Error validateElfTarget(const llvm::object::ELFObjectFileBase &object,
                              const CompilerTargetBinding &target);

} // namespace loom

#endif // LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H
