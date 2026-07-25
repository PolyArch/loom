set(LLVM_LINK_COMPONENTS Support)

add_tablegen(loom-tblgen LOOM
  loom-tblgen.cpp
  OperationRegistryEmitter.cpp
)

set_target_properties(loom-tblgen PROPERTIES FOLDER "Tablegenning")
