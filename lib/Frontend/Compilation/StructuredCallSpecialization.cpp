#include "StructuredCallSpecialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <vector>

namespace loom::frontend::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_call_specialization_invalid: " +
                                     message);
}

bool isLocalDefinition(mlir::LLVM::LLVMFuncOp function) {
  if (!function || function.isExternal() || function.isVarArg())
    return false;
  return function.getLinkage() == mlir::LLVM::Linkage::Internal ||
         function.getLinkage() == mlir::LLVM::Linkage::Private;
}

bool isExactCloneableConstant(mlir::Operation *operation) {
  if (!operation || operation->getNumOperands() != 0 ||
      operation->getNumResults() != 1 || operation->getNumRegions() != 0)
    return false;
  return mlir::isa<mlir::arith::ConstantOp, mlir::LLVM::ConstantOp,
                   mlir::LLVM::ZeroOp, mlir::LLVM::AddressOfOp>(operation);
}

bool sameExactConstant(mlir::Operation *left, mlir::Operation *right) {
  return left && right && left->getName() == right->getName() &&
         left->getResult(0).getType() == right->getResult(0).getType() &&
         left->getAttrDictionary() == right->getAttrDictionary();
}

using KnownArguments =
    llvm::DenseMap<mlir::Operation *, std::vector<mlir::Operation *>>;

mlir::LLVM::LLVMFuncOp owningFunction(mlir::Operation *selection) {
  if (!selection)
    return {};
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(selection))
    return function;
  return selection->getParentOfType<mlir::LLVM::LLVMFuncOp>();
}

mlir::Operation *resolveExactConstant(mlir::Value value,
                                      const KnownArguments &known) {
  if (mlir::Operation *definition = value.getDefiningOp();
      isExactCloneableConstant(definition))
    return definition;
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return nullptr;
  auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      argument.getOwner()->getParentOp());
  if (!function)
    return nullptr;
  auto found = known.find(function.getOperation());
  if (found == known.end() || argument.getArgNumber() >= found->second.size())
    return nullptr;
  return found->second[argument.getArgNumber()];
}

struct UniformBindingProof final {
  mlir::LLVM::LLVMFuncOp function;
  std::vector<mlir::Operation *> arguments;
};

llvm::Expected<UniformBindingProof>
deriveUniformBindings(mlir::ModuleOp module, mlir::Operation *selection) {
  mlir::LLVM::LLVMFuncOp selectedFunction = owningFunction(selection);
  if (!selectedFunction)
    return UniformBindingProof{};

  mlir::SymbolTableCollection symbolTables;
  mlir::SymbolUserMap users(symbolTables, module);
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> functions;
  KnownArguments known;
  for (mlir::LLVM::LLVMFuncOp function :
       module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (!isLocalDefinition(function))
      continue;
    llvm::ArrayRef<mlir::Operation *> symbolUsers =
        users.getUsers(function.getOperation());
    if (symbolUsers.empty() ||
        llvm::any_of(symbolUsers, [&](mlir::Operation *user) {
          auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(user);
          return !call || !call.getCalleeAttr() ||
                 call.getCalleeAttr().getValue() != function.getSymName() ||
                 call.getArgOperands().size() !=
                     function.getFunctionType().getParams().size();
        }))
      continue;
    functions.push_back(function);
    known.try_emplace(function.getOperation(),
                      function.getFunctionType().getParams().size(), nullptr);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::LLVM::LLVMFuncOp function : functions) {
      std::vector<mlir::Operation *> &bindings =
          known.find(function.getOperation())->second;
      llvm::ArrayRef<mlir::Operation *> symbolUsers =
          users.getUsers(function.getOperation());
      for (std::size_t argument = 0; argument < bindings.size(); ++argument) {
        if (bindings[argument])
          continue;
        mlir::Operation *representative = nullptr;
        bool total = true;
        for (mlir::Operation *user : symbolUsers) {
          auto call = llvm::cast<mlir::LLVM::CallOp>(user);
          mlir::Operation *constant =
              resolveExactConstant(call.getArgOperands()[argument], known);
          if (!constant || (representative &&
                            !sameExactConstant(representative, constant))) {
            total = false;
            break;
          }
          representative = constant;
        }
        if (total && representative) {
          bindings[argument] = representative;
          changed = true;
        }
      }
    }
  }

  auto found = known.find(selectedFunction.getOperation());
  if (found == known.end())
    return UniformBindingProof{selectedFunction, {}};
  std::vector<mlir::Operation *> usedBindings = found->second;
  mlir::Block &entry = selectedFunction.getBody().front();
  for (std::size_t argument = 0; argument < usedBindings.size(); ++argument)
    if (entry.getArgument(argument).use_empty())
      usedBindings[argument] = nullptr;
  return UniformBindingProof{selectedFunction, std::move(usedBindings)};
}

} // namespace

llvm::Expected<bool>
hasUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                          mlir::Operation *selection) {
  auto proof = deriveUniformBindings(module, selection);
  if (!proof)
    return proof.takeError();
  return llvm::any_of(proof->arguments,
                      [](mlir::Operation *binding) { return binding; });
}

llvm::Expected<mlir::Operation *>
materializeUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                                  mlir::Operation *selection) {
  auto proof = deriveUniformBindings(module, selection);
  if (!proof)
    return proof.takeError();
  if (!proof->function ||
      llvm::none_of(proof->arguments,
                    [](mlir::Operation *binding) { return binding; }))
    return invalid("selected scope has no uniform exact call arguments");

  constexpr llvm::StringLiteral selectionMarker =
      "loom.call_specialization.selection";
  const bool selectsFunction = selection == proof->function.getOperation();
  if (!selectsFunction) {
    if (!selection || selection->hasAttr(selectionMarker))
      return invalid("selected nested scope has an invalid marker state");
    selection->setAttr(selectionMarker,
                       mlir::UnitAttr::get(module.getContext()));
  }

  mlir::Block &entry = proof->function.getBody().front();
  mlir::OpBuilder builder(&entry, entry.begin());
  for (std::size_t argument = 0; argument < proof->arguments.size();
       ++argument) {
    mlir::Operation *constant = proof->arguments[argument];
    if (!constant)
      continue;
    mlir::IRMapping mapping;
    mlir::Operation *clone = builder.clone(*constant, mapping);
    if (clone->getNumResults() != 1 ||
        clone->getResult(0).getType() != entry.getArgument(argument).getType())
      return invalid("specialized constant changed its exact type");
    entry.getArgument(argument).replaceAllUsesWith(clone->getResult(0));
  }

  mlir::PassManager simplifier =
      mlir::PassManager::on<mlir::LLVM::LLVMFuncOp>(module.getContext());
  simplifier.enableVerifier(true);
  simplifier.addPass(mlir::createSCCPPass());
  simplifier.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(simplifier.run(proof->function.getOperation())))
    return invalid("specialized callable simplification failed");
  if (mlir::failed(mlir::verify(module.getOperation())))
    return invalid("specialized Structured Program does not verify");

  if (selectsFunction)
    return proof->function.getOperation();
  mlir::Operation *specializedSelection = nullptr;
  bool duplicateMarker = false;
  proof->function.walk([&](mlir::Operation *operation) {
    if (!operation->hasAttr(selectionMarker))
      return;
    if (specializedSelection)
      duplicateMarker = true;
    specializedSelection = operation;
  });
  if (duplicateMarker)
    return invalid("selected nested scope marker was duplicated");
  if (!specializedSelection)
    return invalid("selected nested scope was removed by specialization");
  specializedSelection->removeAttr(selectionMarker);
  return specializedSelection;
}

} // namespace loom::frontend::detail
