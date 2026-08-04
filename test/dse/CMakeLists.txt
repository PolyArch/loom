add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

add_executable(loom-dse-candidate-generator-test
  CandidateGeneratorTest.cpp
)

add_executable(loom-dse-plan-test
  PlanTest.cpp
)

target_link_libraries(loom-dse-objective-test PRIVATE
  LoomDSEObjective
  LLVMSupport
)

target_link_libraries(loom-dse-candidate-generator-test PRIVATE
  LoomDSECandidateGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-plan-test PRIVATE
  LoomDSECandidateGenerator
  LLVMSupport
)
