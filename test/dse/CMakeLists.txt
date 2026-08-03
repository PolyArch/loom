add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

target_link_libraries(loom-dse-objective-test PRIVATE
  LoomDSEObjective
  LLVMSupport
)
