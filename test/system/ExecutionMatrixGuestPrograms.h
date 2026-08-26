#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H

#include "llvm/ADT/StringRef.h"

namespace loom::system_test {

llvm::StringRef orderedChannelHostProgramSource();
llvm::StringRef singleInvocationHostProgramSource();
llvm::StringRef pairedInvocationHostProgramSource();
llvm::StringRef spatialInstructionProgramSource();

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H
