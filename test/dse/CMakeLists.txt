add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

add_executable(loom-dse-candidate-generator-test
  CandidateGeneratorTest.cpp
)

target_link_libraries(loom-dse-objective-test PRIVATE
  LoomDSEObjective
  LLVMSupport
)

target_link_libraries(loom-dse-candidate-generator-test PRIVATE
  LoomDSECandidateGenerator
  LLVMSupport
)
