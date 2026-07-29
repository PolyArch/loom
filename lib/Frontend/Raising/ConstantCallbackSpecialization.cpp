#include "Frontend/Raising/StructuredRaising.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "constant_callback_specialization_invalid: " +
                                     message);
}

struct CallbackBinding final {
  unsigned argumentOrdinal;
  llvm::Constant *value;
  llvm::Function *target;
};

bool isExactCallbackFormal(const llvm::Argument &argument,
                           const llvm::Function &target) {
  bool usedAsCallee = false;
  for (const llvm::Instruction &instruction :
       llvm::instructions(argument.getParent())) {
    const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
    if (!call || call->getCalledOperand()->stripPointerCasts() != &argument)
      continue;
    usedAsCallee = true;
    if (call->getFunctionType() != target.getFunctionType() ||
        call->getCallingConv() != target.getCallingConv())
      return false;
  }
  return usedAsCallee;
}

llvm::SmallVector<CallbackBinding, 2> deriveBindings(llvm::CallBase &call,
                                                     llvm::Function &callee) {
  llvm::SmallVector<CallbackBinding, 2> bindings;
  if (callee.isDeclaration() || callee.isVarArg() || callee.getName().empty() ||
      call.arg_size() != callee.arg_size())
    return bindings;

  for (llvm::Argument &argument : callee.args()) {
    llvm::Value *actual = call.getArgOperand(argument.getArgNo());
    auto *constant = llvm::dyn_cast<llvm::Constant>(actual);
    auto *target = llvm::dyn_cast<llvm::Function>(actual->stripPointerCasts());
    if (!constant || !target || target->isDeclaration() ||
        target->getName().empty() || actual->getType() != argument.getType() ||
        !isExactCallbackFormal(argument, *target))
      continue;
    bindings.push_back(CallbackBinding{argument.getArgNo(), constant, target});
  }
  return bindings;
}

std::string bindingKey(llvm::Function &callee,
                       llvm::ArrayRef<CallbackBinding> bindings) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << callee.getName() << '\0';
  for (const CallbackBinding &binding : bindings)
    stream << binding.argumentOrdinal << ':' << binding.target->getName()
           << '\0';
  return key;
}

std::string specializationName(llvm::StringRef key) {
  llvm::MD5 hash;
  hash.update(key);
  llvm::MD5::MD5Result result;
  hash.final(result);
  llvm::SmallString<32> digest;
  llvm::MD5::stringifyResult(result, digest);
  return (llvm::Twine("__loom_callback_specialization_") + digest).str();
}

void appendCalls(llvm::Function &function,
                 llvm::SmallVectorImpl<llvm::CallBase *> &worklist) {
  for (llvm::Instruction &instruction : llvm::instructions(function))
    if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        call && !call->isInlineAsm())
      worklist.push_back(call);
}

llvm::Expected<llvm::Function *> getOrCreateSpecialization(
    llvm::Module &module, llvm::Function &callee,
    llvm::ArrayRef<CallbackBinding> bindings,
    std::map<std::string, llvm::Function *> &specializations,
    llvm::SmallVectorImpl<llvm::CallBase *> &worklist) {
  const std::string key = bindingKey(callee, bindings);
  if (auto found = specializations.find(key); found != specializations.end())
    return found->second;

  const std::string name = specializationName(key);
  if (module.getNamedValue(name))
    return invalid("reserved specialization symbol already exists: " + name);

  llvm::ValueToValueMapTy values;
  llvm::Function *clone = llvm::CloneFunction(&callee, values);
  clone->setName(name);
  clone->setLinkage(llvm::GlobalValue::InternalLinkage);
  clone->setVisibility(llvm::GlobalValue::DefaultVisibility);
  clone->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
  clone->setDSOLocal(true);
  clone->setComdat(nullptr);
  for (const CallbackBinding &binding : bindings) {
    llvm::Argument *argument = clone->getArg(binding.argumentOrdinal);
    if (!argument || argument->getType() != binding.value->getType()) {
      clone->eraseFromParent();
      return invalid("cloned callback formal changed type");
    }
    argument->replaceAllUsesWith(binding.value);
  }
  if (llvm::verifyFunction(*clone)) {
    clone->eraseFromParent();
    return invalid("specialized callback dispatcher failed verification");
  }
  specializations.emplace(key, clone);
  appendCalls(*clone, worklist);
  return clone;
}

} // namespace

llvm::Error specializeExactConstantCallbackCallSites(llvm::Module &module) {
  if (llvm::verifyModule(module))
    return invalid("LLVM module failed verification");

  llvm::SmallVector<llvm::CallBase *, 32> worklist;
  for (llvm::Function &function : module)
    appendCalls(function, worklist);

  std::map<std::string, llvm::Function *> specializations;
  for (std::size_t index = 0; index < worklist.size(); ++index) {
    llvm::CallBase *call = worklist[index];
    llvm::Function *callee = call->getCalledFunction();
    if (!callee)
      continue;
    llvm::SmallVector<CallbackBinding, 2> bindings =
        deriveBindings(*call, *callee);
    if (bindings.empty())
      continue;
    llvm::Expected<llvm::Function *> specialization = getOrCreateSpecialization(
        module, *callee, bindings, specializations, worklist);
    if (!specialization)
      return specialization.takeError();
    call->setCalledFunction(*specialization);
  }

  if (llvm::verifyModule(module))
    return invalid("callback specialization produced invalid LLVM IR");
  return llvm::Error::success();
}

} // namespace loom::raising
